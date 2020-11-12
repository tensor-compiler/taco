#include "test.h"

#include "taco/linalg.h"

using namespace taco;

TEST(linalg, simplest) {
  Matrix<double> B("B", 2, 2, dense, dense);
  Matrix<double> C("C", 2, 2, dense, dense);
  Matrix<double> A("A", 2, 2, dense, dense);

  A = B + C;

  cout << A << endl;

  A.rewrite();
  cout << A.getIndexAssignment();

  ASSERT_TRUE(1);
}

TEST(linalg, matvec_mul) {
  Vector<double> x("x", 2, dense);
  Vector<double> b("b", 2, dense);
  Matrix<double> A("A", 2, 2, dense, dense);

  x = A*b;

  cout << x << endl;

  x.rewrite();
  cout << x.getIndexAssignment();

  ASSERT_TRUE(1);
}

TEST(linalg, vecmat_mul) {
  Vector<double> x("x", 2, dense, false);
  Vector<double> b("b", 2, dense, false);
  Matrix<double> A("A", 2, 2, dense, dense);

  x = b * A;

  cout << x << endl;

  x.rewrite();
  cout << x.getIndexAssignment();

  ASSERT_TRUE(1);
}

TEST(linalg, inner_mul) {
  Scalar<double> x("x");
  Vector<double> b("b", 2, dense, false);
  Vector<double> a("a", 2, dense, true);

  x = b * a;

  cout << x << endl;

  x.rewrite();
  cout << x.getIndexAssignment();

  ASSERT_TRUE(1);
}

TEST(linalg, outer_mul) {
  Matrix<double> X("X", 2, 2, dense, dense);
  Vector<double> b("b", 2, dense, false);
  Vector<double> a("a", 2, dense, true);

  X = a * b;

  cout << X << endl;

  X.rewrite();
  cout << X.getIndexAssignment();

  ASSERT_TRUE(1);
}

TEST(linalg, rowvec_transpose) {
  Vector<double> b("b", 2, dense, false);
  Matrix<double> A("A", 2, 2, dense, dense);
  Scalar<double> a("a");

  a = transpose(transpose(b) * A * b);

  cout << a << endl;

  a.rewrite();
  cout << a.getIndexAssignment();

  ASSERT_TRUE(1);
}