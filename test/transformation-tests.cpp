#include "test.h"
#include "test_tensors.h"

#include "taco/index_notation/schedule.h"
#include "taco/index_notation/transformations.h"
#include "taco/index_notation/index_notation.h"
#include "taco/util/name_generator.h"
#include "taco/tensor.h"

using namespace taco;

static const Dimension n, m, o;
static const Type vectype(Float64, {n});
static const Type mattype(Float64, {n,m});
static const Type tentype(Float64, {n,m,o});

// Sparse vectors
static TensorVar a("a", vectype, Sparse);
static TensorVar b("b", vectype, Sparse);
static TensorVar c("c", vectype, Sparse);
static TensorVar w("w", vectype, dense);

static TensorVar A("A", mattype, Sparse);
static TensorVar B("B", mattype, Sparse);
static TensorVar C("C", mattype, Sparse);

static TensorVar S("S", tentype, Sparse);
static TensorVar T("T", tentype, Sparse);
static TensorVar U("U", tentype, Sparse);

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

TEST_P(precondition, isValid) {
  Transformation transformation = GetParam().transformation;
  IndexStmt invalidStmt = GetParam().invalidStmt;
  ASSERT_FALSE(transformation.isValid(invalidStmt));
}

struct TransformationTest {
  TransformationTest(Transformation transformation, IndexStmt stmt,
                     IndexStmt expected)
      : transformation(transformation), stmt(stmt), expected(expected) {}
  Transformation transformation;
  IndexStmt stmt;
  IndexStmt expected;
};
struct transformation : public TestWithParam<TransformationTest> {};

static ostream &operator<<(ostream& os, const TransformationTest& test) {
  return os << "Transformation: " << test.transformation << endl
            << "Statement:      " << test.stmt           << endl
            << "Expected:       " << test.expected;
}

TEST_P(transformation, isValid) {
  Transformation transformation = GetParam().transformation;
  IndexStmt stmt = GetParam().stmt;
  string reason;
  ASSERT_TRUE(transformation.isValid(stmt, &reason))
      << transformation << " in " << stmt << endl << reason;
}

TEST_P(transformation, apply) {
  Transformation transformation = GetParam().transformation;
  IndexStmt stmt = GetParam().stmt;
  IndexStmt expected = GetParam().expected;
  ASSERT_NOTATION_EQ(expected, transformation.apply(stmt));
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
         PreconditionTest(Reorder(i,j),
                          forall(i,
                                 forall(j,
                                        forall(k,
                                               S(i,j,k) = T(i,j,k)
                                               )))
                          )
         )
);

INSTANTIATE_TEST_CASE_P(reorder, transformation,
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
                            )
         )
);

//TEST(transformations, reorder_ijk) {
//  auto forallijk = forall(i,
//                          forall(j,
//                                 forall(k,
//                                        S(i,j,k) = T(i,j,k)
//                                        )));
//  Reorder reorderij(i,j);
//  ASSERT_TRUE(reorderij.isValid(forallijk));
//}

static Assignment elmul = (a(i) = b(i) * c(i));

INSTANTIATE_TEST_CASE_P(DISABLED_precompute, transformation,
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
