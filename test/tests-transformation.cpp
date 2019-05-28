#include "test.h"
#include "test_tensors.h"

#include "taco/index_notation/schedule.h"
#include "taco/index_notation/transformations.h"
#include "taco/index_notation/index_notation.h"
#include "taco/util/name_generator.h"
#include "taco/tensor.h"

using namespace taco;

// Temporary hack until dense in format.h is transition from the old system
#include "taco/lower/mode_format_dense.h"
taco::ModeFormat denseNew(std::make_shared<taco::DenseModeFormat>());

static const Dimension n, m, o;
static const Type vectype(Float64, {n});
static const Type mattype(Float64, {n,m});
static const Type tentype(Float64, {n,m,o});

// Sparse vectors
static TensorVar a("a", vectype, Sparse);
static TensorVar b("b", vectype, Sparse);
static TensorVar c("c", vectype, Sparse);
static TensorVar w("w", vectype, denseNew);

static TensorVar A("A", mattype, {Sparse, Sparse});
static TensorVar B("B", mattype, {Sparse, Sparse});
static TensorVar C("C", mattype, {Sparse, Sparse});
static TensorVar D("D", mattype, {denseNew, denseNew});
static TensorVar W("W", mattype, {denseNew, denseNew});

static TensorVar S("S", tentype, Sparse);
static TensorVar T("T", tentype, Sparse);
static TensorVar U("U", tentype, Sparse);
static TensorVar V("V", tentype, {denseNew, denseNew, denseNew});
static TensorVar X("X", tentype, {denseNew, denseNew, denseNew});
static TensorVar Y("Y", tentype, {Sparse, denseNew, denseNew});
static TensorVar Z("Z", tentype, {denseNew, denseNew, Sparse});

static const IndexVar i("i"), iw("iw");
static const IndexVar j("j"), jw("jw");
static const IndexVar k("k"), kw("kw");

struct PreconditionTest {
  PreconditionTest(Transformation transformation, IndexStmt invalidStmt)
      : transformation(transformation), invalidStmt(invalidStmt) {}
  Transformation transformation;
  IndexStmt invalidStmt;
};
struct precondition : public TestWithParam<PreconditionTest> {};

ostream& operator<<(ostream& os, const PreconditionTest& test) {
  return os << "Applying " << test.transformation
            << " to " << test.invalidStmt;
}

TEST_P(precondition, transformations) {
  Transformation transformation = GetParam().transformation;
  IndexStmt invalidStmt = GetParam().invalidStmt;
  IndexStmt transformed = transformation.apply(invalidStmt);
  ASSERT_FALSE(transformed.defined()) << "Got " << transformed;
}

struct TransformationTest {
  TransformationTest(Transformation transformation, IndexStmt stmt,
                     IndexStmt expected)
      : transformation(transformation), stmt(stmt), expected(expected) {}
  Transformation transformation;
  IndexStmt stmt;
  IndexStmt expected;
};
struct apply : public TestWithParam<TransformationTest> {};

static ostream &operator<<(ostream& os, const TransformationTest& test) {
  return os << "Transformation: " << test.transformation << endl
            << "Statement:      " << test.stmt           << endl
            << "Expected:       " << test.expected;
}

TEST_P(apply, transformations) {
  Transformation transformation = GetParam().transformation;
  IndexStmt stmt = GetParam().stmt;
  IndexStmt ex = GetParam().expected;
  string reason;
  IndexStmt ac = transformation.apply(stmt, &reason);
  ASSERT_TRUE(ac.defined()) << reason;
  ASSERT_NOTATION_EQ(ex, ac);
}

INSTANTIATE_TEST_CASE_P(reorder, precondition,
  Values(
         PreconditionTest(Reorder(i,j),
                          a(i) = B(i,j)*c(j)
                          ),
         PreconditionTest(Reorder(i,j),
                          a(i) = sum(j,B(i,j)*c(j))
                          ),
         PreconditionTest(Reorder(i,j),
                          forall(i,
                                 multi(forall(j,
                                              A(i,j) = B(i,j)
                                              ),
                                       c(i) = b(i)
                                       ))
                          ),
         PreconditionTest(Reorder(i,j),
                          sequence(forall(j,
                                          A(i,j) = B(i,j)
                                          ),
                                   forall(k,
                                          A(i,k) += C(i,k)
                                          ))
                          ),
         PreconditionTest(Reorder(i,k),
                          forall(i,
                                 forall(j,
                                        forall(k,
                                               S(i,j,k) = T(i,j,k)
                                               )))
                          )
         )
);

INSTANTIATE_TEST_CASE_P(reorder, apply,
  Values(
         TransformationTest(Reorder(i, j),
                            forall(i,
                                   forall(j,
                                          A(i,j) = B(i,j)
                                          )),
                            forall(j,
                                   forall(i,
                                          A(i,j) = B(i,j)
                                          ))
                            ),
         TransformationTest(Reorder(j, i),
                            forall(i,
                                   forall(j,
                                          A(i,j) = B(i,j)
                                          )),
                            forall(j,
                                   forall(i,
                                          A(i,j) = B(i,j)
                                          ))
                            ),
         TransformationTest(Reorder(i, j),
                            forall(i,
                                   forall(j,
                                          A(i,j) += B(i,j)
                                          )),
                            forall(j,
                                   forall(i,
                                          A(i,j) += B(i,j)
                                          ))
                            ),
         TransformationTest(Reorder(j,k),
                          forall(i,
                                 forall(j,
                                        forall(k,
                                               S(i,j,k) = T(i,j,k)
                                               ))),
                                 forall(i,
                                        forall(k,
                                               forall(j,
                                               S(i,j,k) = T(i,j,k)
                                               )))
                          ),
         TransformationTest(Reorder(i,j),
                          forall(i,
                                 forall(j,
                                        forall(k,
                                               S(i,j,k) = T(i,j,k)
                                               ))),
                          forall(j,
                                 forall(i,
                                        forall(k,
                                               S(i,j,k) = T(i,j,k)
                                               )))
                          )
         )
);

static Assignment elmul = (a(i) = b(i) * c(i));

INSTANTIATE_TEST_CASE_P(precompute, apply,
  Values(
         TransformationTest(Precompute(elmul.getRhs(), i, iw, w),
                            makeConcreteNotation(elmul),
                            where(forall(i,
                                         a(i) = w(i)),
                                  forall(iw,
                                         w(iw) = b(iw) * c(iw)))
                            )
  )
);

INSTANTIATE_TEST_CASE_P(parallelize, precondition,
                        Values(
                                PreconditionTest(Parallelize(i),
                                                 forall(i, a(i) = b(i))
                                ),
                                PreconditionTest(Parallelize(i),
                                                 forall(i, w(i) = a(i) + b(i))
                                ),
                                PreconditionTest(Parallelize(i),
                                                 forall(i, forall(j, w(i) = A(i, j) * B(i, j)))
                                )/*, TODO: add precondition when lowering supports reductions
                                PreconditionTest(Parallelize(j),
                                                 forall(i, forall(j, w(j) = W(i, j)))
                                )*/
                        )
);

INSTANTIATE_TEST_CASE_P(parallelize, apply,
                        Values(
                                TransformationTest(Parallelize(i),
                                                   forall(i, w(i) = b(i)),
                                                   forall(i, w(i) = b(i), {Forall::PARALLELIZE})
                                ),
                                TransformationTest(Parallelize(i),
                                                   forall(i, forall(j, W(i,j) = A(i,j))),
                                                   forall(i, forall(j, W(i,j) = A(i,j)), {Forall::PARALLELIZE})
                                ),
                                TransformationTest(Parallelize(j),
                                                   forall(i, forall(j, W(i,j) = A(i,j))),
                                                   forall(i, forall(j, W(i,j) = A(i,j), {Forall::PARALLELIZE}))
                                )
                        )
);

static
IndexNotationTest topoReorderTest(IndexStmt actual, IndexStmt expected) {
  return IndexNotationTest(reorderLoopsTopologically(actual), expected);
}

INSTANTIATE_TEST_CASE_P(reorderLoopsTopologically, notation,
Values(
       topoReorderTest(forall(i, w(i) = b(i)),
                       forall(i, w(i) = b(i))),

       topoReorderTest(forall(i, w(i) = b(i), {Forall::PARALLELIZE}),
                       forall(i, w(i) = b(i), {Forall::PARALLELIZE})),

       topoReorderTest(forall(i, forall(j, W(i,j) = A(i,j))),
                       forall(i, forall(j, W(i,j) = A(i,j)))),

       topoReorderTest(forall(j, forall(i, W(i,j) = A(i,j))),
                       forall(i, forall(j, W(i,j) = A(i,j)))),

       topoReorderTest(forall(j, forall(i, W(i,j) = D(i,j))),
                       forall(j, forall(i, W(i,j) = D(i,j)))),

       topoReorderTest(forall(i, forall(j, W(j,i) = D(i,j))),
                       forall(i, forall(j, W(j,i) = D(i,j)))),

       topoReorderTest(forall(j, forall(i, A(i,j) = D(i,j))),
                       forall(i, forall(j, A(i,j) = D(i,j)))),

       topoReorderTest(forall(j, forall(i, W(i,j) = D(i,j) + A(i, j))),
                       forall(i, forall(j, W(i,j) = D(i,j) + A(i, j)))),

       topoReorderTest(forall(i, forall(j, forall(k, X(i,j,k) = V(i,j,k)))),
                       forall(i, forall(j, forall(k, X(i,j,k) = V(i,j,k))))),

       topoReorderTest(forall(k, forall(j, forall(i, X(i,j,k) = Y(i,j,k)))),
                       forall(i, forall(k, forall(j, X(i,j,k) = Y(i,j,k))))),

       topoReorderTest(forall(k, forall(j, forall(i, X(i,j,k) = Z(i,j,k)))),
                       forall(j, forall(i, forall(k, X(i,j,k) = Z(i,j,k)))))
  )
);

/*
TEST(schedule, workspace_spmspm) {
  TensorBase A("A", Float(64), {3,3}, Format({Dense,Sparse}));
  TensorBase B = d33a("B", Format({Dense,Sparse}));
  TensorBase C = d33b("C", Format({Dense,Sparse}));
  B.pack();
  C.pack();

  IndexVar i, j, k;
  IndexExpr matmul = B(i,k) * C(k,j);
  A(i,j) = matmul;

  A.evaluate();

  std::cout << A << std::endl;
  Tensor<double> E("e", {3,3}, Format({Dense,Sparse}));
  E.insert({2,0}, 30.0);
  E.insert({2,1}, 180.0);
  E.pack();
  ASSERT_TENSOR_EQ(E,A);
}
*/
