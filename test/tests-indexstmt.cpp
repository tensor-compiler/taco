#include "test.h"
#include "test_tensors.h"
#include "taco/tensor.h"
#include "taco/index_notation/index_notation.h"

using namespace taco;
const IndexVar i("i"), j("j"), k("k");

TEST(indexstmt, assignment) {
  Type t(type<double>(), {3});
  TensorVar a("a", t), b("b", t), c("c", t);

  IndexStmt stmt = a(i) = b(i) + c(i);
  ASSERT_TRUE(isa<Assignment>(stmt));
  Assignment assignment = to<Assignment>(stmt);
  ASSERT_TRUE(equals(a(i), assignment.getLhs()));
  ASSERT_TRUE(equals(b(i) + c(i), assignment.getRhs()));
  ASSERT_EQ(IndexExpr(), assignment.getOperator());
}

TEST(indexstmt, forall) {
  Type t(type<double>(), {3});
  TensorVar a("a", t), b("b", t), c("c", t);

  IndexStmt stmt = forall(i, a(i) = b(i) + c(i));
  ASSERT_TRUE(isa<Forall>(stmt));
  Forall forallstmt = to<Forall>(stmt);
  ASSERT_EQ(i, forallstmt.getIndexVar());
  ASSERT_TRUE(equals(a(i) = b(i) + c(i), forallstmt.getStmt()));
  ASSERT_TRUE(equals(forall(i, a(i) = b(i) + c(i)), forallstmt));
}

TEST(indexstmt, where) {
  Type t(type<double>(), {3});
  TensorVar a("a", t, Sparse), b("b", t, Sparse), c("c", t, Sparse);
  TensorVar w("w", t, Dense);

  IndexStmt stmt = where(forall(i, a(i)=w(i)*c(i)), forall(i, w(i)=b(i)));
  ASSERT_TRUE(isa<Where>(stmt));
  Where wherestmt = to<Where>(stmt);
  ASSERT_TRUE(equals(forall(i, a(i)=w(i)*c(i)), wherestmt.getConsumer()));
  ASSERT_TRUE(equals(forall(i, w(i)=b(i)), wherestmt.getProducer()));
  ASSERT_TRUE(equals(where(forall(i, a(i)=w(i)*c(i)), forall(i, w(i)=b(i))),
                     wherestmt));
}

TEST(indexstmt, multi) {
  Type t(type<double>(), {3});
  TensorVar a("a", t, Sparse), b("b", t, Sparse), c("c", t, Sparse);

  IndexStmt stmt = multi(a(i)=c(i), b(i)=c(i));
  ASSERT_TRUE(isa<Multi>(stmt));
  Multi multistmt = to<Multi>(stmt);
  ASSERT_TRUE(equals(multistmt.getStmt1(), a(i) = c(i)));
  ASSERT_TRUE(equals(multistmt.getStmt2(), b(i) = c(i)));
  ASSERT_TRUE(equals(multistmt, multi(a(i)=c(i), b(i)=c(i))));
}

TEST(indexstmt, sequence) {
  Type t(type<double>(), {3});
  TensorVar a("a", t, Sparse), b("b", t, Sparse), c("c", t, Sparse);

  IndexStmt stmt = sequence(a(i) = b(i), a(i) += c(i));
  ASSERT_TRUE(isa<Sequence>(stmt));
  Sequence sequencestmt = to<Sequence>(stmt);
  ASSERT_TRUE(equals(a(i) = b(i), sequencestmt.getDefinition()));
  ASSERT_TRUE(equals(a(i) += c(i), sequencestmt.getMutation()));
  ASSERT_TRUE(equals(sequence(a(i) = b(i), a(i) += c(i)),
                     sequencestmt));
}

TEST(indexstmt, spmm) {
  Type t(type<double>(), {3,3});
  TensorVar A("A", t, Sparse), B("B", t, Sparse), C("C", t, Sparse);
  TensorVar w("w", Type(type<double>(),{3}), Dense);

  auto spmm = forall(i,
                     forall(k,
                            where(forall(j, A(i,j) = w(j)),
                                  forall(j,   w(j) += B(i,k)*C(k,j))
                                  )
                            )
                     );
}



