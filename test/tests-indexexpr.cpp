#include "test.h"
#include "test_tensors.h"
#include "taco/tensor.h"
#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/index_notation_nodes.h"

using namespace taco;
const IndexVar i("i"), j("j"), k("k");

Type vectype(type<double>(), {3});
TensorVar a("a", vectype), b("b", vectype), c("c", vectype);

TEST(indexexpr, access) {
  IndexExpr expr = b(i);
  ASSERT_TRUE(isa<Access>(expr));
  ASSERT_TRUE(isa<AccessNode>(expr.ptr));
  Access access = to<Access>(expr);
  ASSERT_EQ(access.getTensorVar(), b);
  ASSERT_NE(access.getTensorVar(), c);
  ASSERT_EQ(access.getIndexVars()[0], i);
  ASSERT_NE(access.getIndexVars()[0], j);
}

TEST(indexexpr, literal) {
  IndexExpr expr = 20;
  ASSERT_TRUE(isa<Literal>(expr));
  ASSERT_TRUE(isa<LiteralNode>(expr.ptr));
  Literal literal = to<Literal>(expr);
  ASSERT_EQ(type<int>(), literal.getDataType());
  ASSERT_EQ(20, literal.getVal<int>());
}

TEST(indexexpr, neg) {
  IndexExpr expr = -b(i);
  ASSERT_TRUE(isa<Neg>(expr));
  ASSERT_TRUE(isa<NegNode>(expr.ptr));
  Neg neg = to<Neg>(expr);
  ASSERT_TRUE(equals(neg.getA(), b(i)));
}

TEST(indexexpr, add) {
  IndexExpr expr = b(i) + c(i);
  ASSERT_TRUE(isa<Add>(expr));
  ASSERT_TRUE(isa<AddNode>(expr.ptr));
  Add add = to<Add>(expr);
  ASSERT_TRUE(equals(add.getA(), b(i)));
  ASSERT_TRUE(equals(add.getB(), c(i)));
}

TEST(indexexpr, sub) {
  IndexExpr expr = b(i) - c(i);
  ASSERT_TRUE(isa<Sub>(expr));
  ASSERT_TRUE(isa<SubNode>(expr.ptr));
  Sub sub = to<Sub>(expr);
  ASSERT_TRUE(equals(sub.getA(), b(i)));
  ASSERT_TRUE(equals(sub.getB(), c(i)));
}

TEST(indexexpr, mul) {
  IndexExpr expr = b(i) * c(i);
  ASSERT_TRUE(isa<Mul>(expr));
  ASSERT_TRUE(isa<MulNode>(expr.ptr));
  Mul mul = to<Mul>(expr);
  ASSERT_TRUE(equals(mul.getA(), b(i)));
  ASSERT_TRUE(equals(mul.getB(), c(i)));
}

TEST(indexexpr, div) {
  IndexExpr expr = b(i) / 2;
  ASSERT_TRUE(isa<Div>(expr));
  ASSERT_TRUE(isa<DivNode>(expr.ptr));
  Div div = to<Div>(expr);
  ASSERT_TRUE(equals(div.getA(), b(i)));
  ASSERT_TRUE(equals(div.getB(), Literal(2)));
}
