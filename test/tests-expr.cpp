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

TEST(expr, accumulate) {
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

  ASSERT_EQ(neg, simplify(neg, {}));
  ASSERT_EQ(IndexExpr(), simplify(neg, {Bex}));
}

TEST(expr, simplify_elmul) {
  Type mat(type<double>(), {3,3});
  TensorVar B("B", mat), C("C", mat);
  IndexVar i("i"), j("j");

  Access Bex = B(i,j);
  Access Cex = C(i,j);
  IndexExpr mul = Bex * Cex;

  ASSERT_EQ(mul, simplify(mul, {}));
  ASSERT_EQ(IndexExpr(), simplify(mul, {Bex}));
  ASSERT_EQ(IndexExpr(), simplify(mul, {Cex}));
  ASSERT_EQ(IndexExpr(), simplify(mul, {Bex,Cex}));
}

TEST(expr, simplify_add) {
  Type mat(type<double>(), {3,3});
  TensorVar B("B", mat), C("C", mat);
  IndexVar i("i"), j("j");

  Access Bex = B(i,j);
  Access Cex = C(i,j);
  IndexExpr mul = Bex + Cex;

  ASSERT_EQ(mul, simplify(mul, {}));
  ASSERT_EQ(Cex, simplify(mul, {Bex}));
  ASSERT_EQ(Bex, simplify(mul, {Cex}));
  ASSERT_EQ(IndexExpr(), simplify(mul, {Bex,Cex}));
}

TEST(expr, simplify_addmul) {
  Type mat(type<double>(), {3,3});
  TensorVar B("B", mat), C("C", mat), D("D", mat);
  IndexVar i("i"), j("j");

  Access Bex = B(i,j);
  Access Cex = C(i,j);
  Access Dex = D(i,j);
  IndexExpr addmul = (Bex + Cex) * Dex;

  ASSERT_EQ(addmul, simplify(addmul, {}));
  ASSERT_NOTATION_EQ(Cex * Dex, simplify(addmul, {Bex}));
  ASSERT_NOTATION_EQ(Bex * Dex, simplify(addmul, {Cex}));
  ASSERT_NOTATION_EQ(IndexExpr(), simplify(addmul, {Dex}));
  ASSERT_NOTATION_EQ(IndexExpr(), simplify(addmul, {Bex, Dex}));
  ASSERT_NOTATION_EQ(IndexExpr(), simplify(addmul, {Cex, Dex}));
  ASSERT_NOTATION_EQ(IndexExpr(), simplify(addmul, {Bex, Cex, Dex}));
}

TEST(expr, simplify_muladd) {
  Type mat(type<double>(), {3,3});
  TensorVar B("B", mat), C("C", mat), D("D", mat);
  IndexVar i("i"), j("j");

  Access Bex = B(i,j);
  Access Cex = C(i,j);
  Access Dex = D(i,j);
  IndexExpr addmul = (Bex * Cex) + Dex;

  ASSERT_EQ(addmul, simplify(addmul, {}));
  ASSERT_NOTATION_EQ(Dex, simplify(addmul, {Bex}));
  ASSERT_NOTATION_EQ(Dex, simplify(addmul, {Cex}));
  ASSERT_NOTATION_EQ(Bex * Cex, simplify(addmul, {Dex}));
  ASSERT_NOTATION_EQ(IndexExpr(), simplify(addmul, {Bex, Dex}));
  ASSERT_NOTATION_EQ(IndexExpr(), simplify(addmul, {Cex, Dex}));
  ASSERT_NOTATION_EQ(IndexExpr(), simplify(addmul, {Bex, Cex, Dex}));
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
