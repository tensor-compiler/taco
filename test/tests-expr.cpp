#include "test.h"
#include "test_tensors.h"
#include "taco/tensor.h"

using namespace taco;

const IndexVar i("i"), j("j"), k("k");

TEST(expr, repeated_operand) {
  Tensor<double> a("a", {8}, Format({Sparse}, {0}));
  Tensor<double> b = d8b("b", Format({Sparse}, {0}));
  b.pack();
  a(i) = b(i) + b(i);
  a.evaluate();

  Tensor<double> expected("a", {8}, Format({Sparse}, {0}));
  expected.insert({0}, 20.0);
  expected.insert({2}, 40.0);
  expected.insert({3}, 60.0);
  expected.pack();
  ASSERT_TRUE(equals(expected,a));
}

TEST(expr, DISABLED_accumulate) {
  Tensor<double> a = d8a("a2", Format({Dense}, {0}));
  Tensor<double> b = d8b("b", Format({Sparse}, {0}));
  a.pack();
  b.pack();
  a(i) += b(i);
  a.evaluate();

  Tensor<double> expected("e", {8}, Format({Dense}, {0}));
  expected.insert({0}, 11.0);
  expected.insert({1}, 2.0);
  expected.insert({2}, 23.0);
  expected.insert({3}, 30.0);
  expected.insert({5}, 4.0);
  expected.pack();
  ASSERT_TRUE(equals(expected,a)) << endl << expected << endl << endl << a;
}

TEST(expr, sub) {
  Tensor<double> a("a", {2}, Format({Sparse}, {0}));
  Tensor<double> b("b", {2}, Format({Sparse}, {0}));
  Tensor<double> c("c", {2}, Format({Dense}, {0}));
  Tensor<double> expected("c_soln", {2}, Format({Dense}, {0}));

  a.insert({0}, 1.0);
  a.pack();

  b.insert({1}, 1.0);
  b.pack();

  expected.insert({0}, 1.0);
  expected.insert({1}, -1.0);
  expected.pack();

  c(i) = a(i) - b(i);
  c.evaluate();
  ASSERT_TRUE(equals(expected, c));
}

TEST(expr, simplify_neg) {
  Type mat(type<double>(), {3,3});
  TensorVar B("B", mat);
  IndexVar i("i"), j("j");

  Access Bex = B(i,j);
  IndexExpr neg = -Bex;

  ASSERT_EQ(neg, zero(neg, {}));
  ASSERT_EQ(IndexExpr(), zero(neg, {Bex}));
}

TEST(expr, simplify_elmul) {
  Type mat(type<double>(), {3,3});
  TensorVar B("B", mat), C("C", mat);
  IndexVar i("i"), j("j");

  Access Bex = B(i,j);
  Access Cex = C(i,j);
  IndexExpr mul = Bex * Cex;

  ASSERT_EQ(mul, zero(mul, {}));
  ASSERT_EQ(IndexExpr(), zero(mul, {Bex}));
  ASSERT_EQ(IndexExpr(), zero(mul, {Cex}));
  ASSERT_EQ(IndexExpr(), zero(mul, {Bex,Cex}));
}

TEST(expr, simplify_add) {
  Type mat(type<double>(), {3,3});
  TensorVar B("B", mat), C("C", mat);
  IndexVar i("i"), j("j");

  Access Bex = B(i,j);
  Access Cex = C(i,j);
  IndexExpr mul = Bex + Cex;

  ASSERT_EQ(mul, zero(mul, {}));
  ASSERT_EQ(Cex, zero(mul, {Bex}));
  ASSERT_EQ(Bex, zero(mul, {Cex}));
  ASSERT_EQ(IndexExpr(), zero(mul, {Bex,Cex}));
}

TEST(expr, simplify_addmul) {
  Type mat(type<double>(), {3,3});
  TensorVar B("B", mat), C("C", mat), D("D", mat);
  IndexVar i("i"), j("j");

  Access Bex = B(i,j);
  Access Cex = C(i,j);
  Access Dex = D(i,j);
  IndexExpr addmul = (Bex + Cex) * Dex;

  ASSERT_EQ(addmul, zero(addmul, {}));
  ASSERT_NOTATION_EQ(Cex * Dex, zero(addmul, {Bex}));
  ASSERT_NOTATION_EQ(Bex * Dex, zero(addmul, {Cex}));
  ASSERT_NOTATION_EQ(IndexExpr(), zero(addmul, {Dex}));
  ASSERT_NOTATION_EQ(IndexExpr(), zero(addmul, {Bex, Dex}));
  ASSERT_NOTATION_EQ(IndexExpr(), zero(addmul, {Cex, Dex}));
  ASSERT_NOTATION_EQ(IndexExpr(), zero(addmul, {Bex, Cex, Dex}));
}

TEST(expr, simplify_muladd) {
  Type mat(type<double>(), {3,3});
  TensorVar B("B", mat), C("C", mat), D("D", mat);
  IndexVar i("i"), j("j");

  Access Bex = B(i,j);
  Access Cex = C(i,j);
  Access Dex = D(i,j);
  IndexExpr addmul = (Bex * Cex) + Dex;

  ASSERT_EQ(addmul, zero(addmul, {}));
  ASSERT_NOTATION_EQ(Dex, zero(addmul, {Bex}));
  ASSERT_NOTATION_EQ(Dex, zero(addmul, {Cex}));
  ASSERT_NOTATION_EQ(Bex * Cex, zero(addmul, {Dex}));
  ASSERT_NOTATION_EQ(IndexExpr(), zero(addmul, {Bex, Dex}));
  ASSERT_NOTATION_EQ(IndexExpr(), zero(addmul, {Cex, Dex}));
  ASSERT_NOTATION_EQ(IndexExpr(), zero(addmul, {Bex, Cex, Dex}));
}

TEST(expr, scalarops) {
  TensorVar a("a", Float64), b("b", Float64), c("c", Float64);
  // check that scalar operations compile
  a = -b;
  b + c;
  b - c;
  b * c;
  b / c;
}

TEST(expr, redefine) {
  Tensor<double> a;
  a = 40.0;
  a.evaluate();
  ASSERT_EQ(a.begin()->second, 40.0);

  a = 42.0;
  a.evaluate();
  ASSERT_EQ(a.begin()->second, 42.0);
}
