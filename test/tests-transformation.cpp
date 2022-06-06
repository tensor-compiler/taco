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

static TensorVar a("a", vectype, compressed);
static TensorVar b("b", vectype, compressed);
static TensorVar c("c", vectype, compressed);
static TensorVar w("w", vectype, dense);

static TensorVar A("A", mattype, {dense, compressed});
static TensorVar B("B", mattype, {dense, compressed});
static TensorVar C("C", mattype, {dense, compressed});
static TensorVar D("D", mattype, {compressed, compressed});
static TensorVar E("E", mattype, {compressed, compressed});
static TensorVar F("F", mattype, {compressed, compressed});
static TensorVar G("D", mattype, {dense, dense});
static TensorVar W("W", mattype, {dense, dense});

static TensorVar S("S", tentype, compressed);
static TensorVar T("T", tentype, compressed);
static TensorVar U("U", tentype, compressed);
static TensorVar V("V", tentype, {dense, dense, dense});
static TensorVar X("X", tentype, {dense, dense, dense});
static TensorVar Y("Y", tentype, {compressed, dense, dense});
static TensorVar Z("Z", tentype, {dense, dense, compressed});
static TensorVar Q("Q", tentype, Format({dense, dense, dense}, {1, 2, 0}));
static TensorVar R("R", tentype, Format({dense, dense, dense}, {1, 2, 0}));

static const IndexVar i("i"), iw("iw");
static const IndexVar j("j"), jw("jw");
static const IndexVar k("k"), kw("kw");

namespace test {

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
                          ),
         TransformationTest(Reorder({j, i, k}),
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
         ),
         TransformationTest(Reorder({k, j, i}),
                            forall(i,
                                   forall(j,
                                          forall(k,
                                                 S(i,j,k) = T(i,j,k)
                                          ))),
                            forall(k,
                                   forall(j,
                                          forall(i,
                                                 S(i,j,k) = T(i,j,k)
                                          )))
         ))
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

INSTANTIATE_TEST_CASE_P(parallelize, precondition, Values(
  PreconditionTest(Parallelize(i),
                   forall(i, a(i) = b(i))),
  PreconditionTest(Parallelize(i),
                   forall(i, w(i) = a(i) + b(i)) ),
  PreconditionTest(Parallelize(i),
                   forall(i, forall(j, W(i, j) = D(i, j) * E(i, j)))),
  PreconditionTest(Parallelize(i),
                   forall(i, forall(j, A(i, j) = W(i, j)))),
  PreconditionTest(Parallelize(i),
                   forall(i, forall(j, E(i, j) = W(i, j)))),
  PreconditionTest(Parallelize(i),
                   forall(i, forall(j, w(j) = W(i, j)))))
);

INSTANTIATE_TEST_CASE_P(parallelize, apply,
                        Values(
                                TransformationTest(Parallelize(i),
                                                   forall(i, w(i) = b(i)),
                                                   forall(i, w(i) = b(i), MergeStrategy::TwoFinger, ParallelUnit::DefaultUnit, OutputRaceStrategy::NoRaces)
                                ),
                                TransformationTest(Parallelize(i),
                                                   forall(i, forall(j, W(i,j) = A(i,j))),
                                                   forall(i, forall(j, W(i,j) = A(i,j)), MergeStrategy::TwoFinger, ParallelUnit::DefaultUnit, OutputRaceStrategy::NoRaces)
                                ),
                                TransformationTest(Parallelize(j),
                                                   forall(i, forall(j, W(i,j) = A(i,j))),
                                                   forall(i, forall(j, W(i,j) = A(i,j), MergeStrategy::TwoFinger, ParallelUnit::DefaultUnit, OutputRaceStrategy::NoRaces))
                                )
                        )
);


struct reorderLoopsTopologically : public TestWithParam<NotationTest> {};

TEST_P(reorderLoopsTopologically, test) {
  IndexStmt actual = taco::reorderLoopsTopologically(GetParam().actual);
  ASSERT_NOTATION_EQ(GetParam().expected, actual);
}

INSTANTIATE_TEST_CASE_P(misc, reorderLoopsTopologically, Values(
  NotationTest(forall(i, w(i) = b(i)),
                  forall(i, w(i) = b(i))),

  NotationTest(forall(i, w(i) = b(i), MergeStrategy::TwoFinger, ParallelUnit::DefaultUnit, OutputRaceStrategy::NoRaces),
                  forall(i, w(i) = b(i), MergeStrategy::TwoFinger, ParallelUnit::DefaultUnit, OutputRaceStrategy::NoRaces)),

  NotationTest(forall(i, forall(j, W(i,j) = A(i,j))),
                  forall(i, forall(j, W(i,j) = A(i,j)))),

  NotationTest(forall(j, forall(i, W(i,j) = A(i,j))),
                  forall(i, forall(j, W(i,j) = A(i,j)))),

  NotationTest(forall(j, forall(i, W(i,j) = G(i,j))),
                  forall(i, forall(j, W(i,j) = G(i,j)))),

  NotationTest(forall(i, forall(j, W(j,i) = G(i,j))),
                  forall(i, forall(j, W(j,i) = G(i,j)))),

  NotationTest(forall(j, forall(i, A(i,j) = G(i,j))),
                  forall(i, forall(j, A(i,j) = G(i,j)))),

  NotationTest(forall(j, forall(i, W(i,j) = G(i,j) + A(i, j))),
                  forall(i, forall(j, W(i,j) = G(i,j) + A(i, j)))),

  NotationTest(forall(i, forall(j, forall(k, X(i,j,k) = V(i,j,k)))),
                  forall(i, forall(j, forall(k, X(i,j,k) = V(i,j,k))))),

  NotationTest(forall(k, forall(j, forall(i, X(i,j,k) = Y(i,j,k)))),
                  forall(i, forall(j, forall(k, X(i,j,k) = Y(i,j,k))))),

  NotationTest(forall(k, forall(j, forall(i, X(i,j,k) = Z(i,j,k)))),
                  forall(i, forall(j, forall(k, X(i,j,k) = Z(i,j,k))))),

  NotationTest(forall(i, forall(j, forall(k, Q(i,j,k) = R(i,j,k)))),
                  forall(j, forall(k, forall(i, Q(i,j,k) = R(i,j,k))))),

  NotationTest(forall(i,
                      forall(j,
                             forall(k,
                                    A(i,j) += B(i,k) * C(k,j)))),
               forall(i,
                      forall(k,
                             forall(j,
                                    A(i,j) += B(i,k) * C(k,j)))))
));


struct insertTemporaries : public TestWithParam<NotationTest> {};

TEST_P(insertTemporaries, test) {
  IndexStmt actual = taco::insertTemporaries(GetParam().actual);
  ASSERT_NOTATION_EQ(GetParam().expected, actual);
}

INSTANTIATE_TEST_CASE_P(spmm, insertTemporaries, Values(
  NotationTest(forall(i,
                      forall(k,
                             forall(j,
                                    A(i,j) += B(i,k) * C(k,j)))),
               forall(i,
                      where(forall(j,
                                   A(i,j) = w(j)),
                            forall(k,
                                   forall(j,
                                          w(j) += B(i,k) * C(k,j))))))
));

TEST(schedule, workspace_spmspm) {
  TensorBase A("A", Float(64), {3,3}, Format({dense,compressed}));
  TensorBase B = d33a("B", Format({dense,compressed}));
  TensorBase C = d33b("C", Format({dense,compressed}));
  B.pack();
  C.pack();

  IndexVar i, j, k;
  IndexExpr matmul = B(i,k) * C(k,j);
  A(i,j) = matmul;

  A.evaluate();

  Tensor<double> E("e", {3,3}, Format({dense,compressed}));
  E.insert({2,0}, 30.0);
  E.insert({2,1}, 180.0);
  E.pack();
  ASSERT_TENSOR_EQ(E,A);
}

}
