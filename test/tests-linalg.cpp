#include "test.h"

#include "taco/linalg.h"

using namespace taco;

TEST(linalg, reassignment) {
  Matrix<double> A("A", {2,2});
  Matrix<double> B1("B1", {2,2});
  Matrix<double> B2("B2", {2,2});
  Matrix<double> B3("B3", {2,2});
  Matrix<double> C1("C1", {2,2});
  Matrix<double> C2("C2", {2,2});
  Matrix<double> C3("C3", {2,2});

  A = B1 * C1;

  IndexVar i,j,k;
  A(i,j) = B2(i,k) * C2(k,j);

  A = B3 * C3;
}

TEST(linalg, tensor_comparison) {
  Matrix<double> A("A", {2,2});
  Tensor<double> B("B", {2,2});

  A(0,0) = 1;
  A(1,1) = 1;

  B(0,0) = 1;
  B(1,1) = 1;

  ASSERT_TENSOR_EQ(A,B);
}

TEST(linalg, matrix_constructors) {
  Matrix<double> A("A");
  Matrix<double> B("B", {2, 2});
  Matrix<double> C("C", 2, 2, dense, dense);
  Matrix<double> D("D", 2, 2);
  Matrix<double> E("E", 2, 2, {dense, dense});
  Matrix<double> F("F", {2, 2}, {dense, dense});

  Vector<double> a("a");
  Vector<double> b("b", 2, false);
  Vector<double> c("c", 2, dense);
  Vector<double> d("d", 2, {dense});
}

TEST(linalg, matmul_index_expr) {
  Tensor<double> B("B", {2,2});
  Matrix<double> C("C", 2, 2, dense, dense);
  Matrix<double> A("A", 2, 2, dense, dense);

  B(0,0) = 2;
  B(1,1) = 1;
  B(0,1) = 2;
  C(0,0) = 2;
  C(1,1) = 2;

  IndexVar i, j, k;
  A(i,j) = B(i,k) * C(k,j);

  ASSERT_EQ((double) A(0,0), 4);
  ASSERT_EQ((double) A(0,1), 4);
  ASSERT_EQ((double) A(1,0), 0);
  ASSERT_EQ((double) A(1,1), 2);
}

TEST(linalg, vecmat_mul_index_expr) {
  Vector<double> x("x", 2, dense, false);
  Vector<double> b("b", 2, dense, false);
  Matrix<double> A("A", 2, 2, dense, dense);

  b(0) = 3;
  b(1) = -2;

  A(0,0) = 5;
  A(0,1) = 2;
  A(1,0) = -1;

  // Should be [17, 6]
  IndexVar i, j;
  x(i) = b(j) * A(j,i);

  ASSERT_EQ((double) x(0), 17);
  ASSERT_EQ((double) x(1), 6);

  cout << x << endl;

  cout << x.getIndexAssignment();
}


TEST(linalg, inner_mul_index_expr) {
  Scalar<double> x("x", true);
  Vector<double> b("b", 2, dense, false);
  Vector<double> a("a", 2, dense, true);

  b(0) = 2;
  b(1) = 3;

  a(0) = -3;
  a(1) = 5;

  IndexVar i;
  x = b(i) * a(i);

  // Should be 9
  cout << x << endl;

  cout << x.getIndexAssignment();

  ASSERT_EQ((double) x, 9);
}

TEST(linalg, matmul) {
  Matrix<double> B("B", 2, 2, dense, dense);
  Matrix<double> C("C", 2, 2, dense, dense);
  Matrix<double> A("A", 2, 2, dense, dense);

  B(0,0) = 2;
  B(1,1) = 1;
  B(0,1) = 2;
  C(0,0) = 2;
  C(1,1) = 2;

  A = B * C;

  ASSERT_EQ((double) A(0,0), 4);
  ASSERT_EQ((double) A(0,1), 4);
  ASSERT_EQ((double) A(1,0), 0);
  ASSERT_EQ((double) A(1,1), 2);

  // Equivalent Tensor API computation
  Tensor<double> tB("B", {2, 2}, dense);
  Tensor<double> tC("C", {2, 2}, dense);
  Tensor<double> tA("A", {2, 2}, dense);

  tB(0,0) = 2;
  tB(1,1) = 1;
  tB(0,1) = 2;
  tC(0,0) = 2;
  tC(1,1) = 2;

  IndexVar i,j,k;
  tA(i,j) = tB(i,k) * tC(k,j);

  ASSERT_TENSOR_EQ(A,tA);
}

TEST(linalg, matmat_add) {
  Matrix<double> B("B", 2, 2, dense, dense);
  Matrix<double> C("C", 2, 2, dense, dense);
  Matrix<double> A("A", 2, 2, dense, dense);

  B(0,0) = 1;
  B(1,1) = 4;

  C(0,1) = 2;
  C(1,0) = 3;

  A = B + C;

  ASSERT_EQ((double) A(0,0), 1);
  ASSERT_EQ((double) A(0,1), 2);
  ASSERT_EQ((double) A(1,0), 3);
  ASSERT_EQ((double) A(1,1), 4);
}

TEST(linalg, matvec_mul) {
  Vector<double> x("x", 2, dense);
  Vector<double> b("b", 2, dense);
  Matrix<double> A("A", 2, 2, dense, dense);

  b(0) = 2;
  b(1) = 1;

  A(0,0) = 1;
  A(0,1) = 3;
  A(1,1) = 2;

  x = A*b;

  ASSERT_EQ((double) x(0), 5);
  ASSERT_EQ((double) x(1), 2);

  // Should be [5,2]
  cout << x << endl;

  cout << x.getIndexAssignment();
}

TEST(linalg, vecmat_mul) {
  Vector<double> x("x", 2, dense, false);
  Vector<double> b("b", 2, dense, false);
  Matrix<double> A("A", 2, 2, dense, dense);

  b(0) = 3;
  b(1) = -2;

  A(0,0) = 5;
  A(0,1) = 2;
  A(1,0) = -1;

  // Should be [17, 6]
  x = b * A;

  ASSERT_EQ((double) x(0), 17);
  ASSERT_EQ((double) x(1), 6);

  cout << x << endl;

  cout << x.getIndexAssignment();
}

TEST(linalg, inner_mul) {
  Scalar<double> x("x", true);
  Vector<double> b("b", 2, dense, false);
  Vector<double> a("a", 2, dense, true);

  b(0) = 2;
  b(1) = 3;

  a(0) = -3;
  a(1) = 5;

  x = b * a;

  // Should be 9
  cout << x << endl;

  cout << x.getIndexAssignment();

  ASSERT_EQ((double) x, 9);
}

TEST(linalg, outer_mul) {
  Matrix<double> X("X", 2, 2, dense, dense);
  Vector<double> b("b", 2, dense, false);
  Vector<double> a("a", 2, dense, true);

  b(0) = 2;
  b(1) = 3;

  a(0) = -3;
  a(1) = 5;

  X = a * b;

  // Should be [-6,-9,10,15]
  cout << X << endl;

  cout << X.getIndexAssignment();

  cout << X;

  ASSERT_TRUE(1);
}

/* TEST(linalg, rowvec_transpose) { */
/*   Vector<double> b("b", 2, dense, false); */
/*   Matrix<double> A("A", 2, 2, dense, dense); */
/*   Scalar<double> a("a", true); */

/*   b(0) = 2; */
/*   b(1) = 5; */

/*   A(0,0) = 1; */
/*   A(0,1) = 2; */
/*   A(1,1) = 4; */

/*   a = transpose(transpose(b) * A * b); */

/*   // Should be 124 */
/*   cout << a << endl; */

/*   cout << a.getIndexAssignment(); */

/*   ASSERT_TRUE(1); */
/* } */

TEST(linalg, compound_expr_elemmul_elemadd) {
  Matrix<double> A("A", 2, 2, dense, dense);
  Matrix<double> B("B", 2, 2, dense, dense);
  Matrix<double> C("C", 2, 2, dense, dense);
  Matrix<double> D("D", 2, 2, dense, dense);

  Tensor<double> tA("A", {2,2}, dense);
  Tensor<double> tB("B", {2,2}, dense);
  Tensor<double> tC("C", {2,2}, dense);
  Tensor<double> tD("D", {2,2}, dense);

  A(0,0) = 1;
  A(0,1) = 2;
  A(0,2) = 3;

  tA(0,0) = 1;
  tA(0,1) = 2;
  tA(0,2) = 3;

  D(0,0) = 2;
  D(0,1) = 3;
  D(0,2) = 4;

  tD(0,0) = 2;
  tD(0,1) = 3;
  tD(0,2) = 4;

  A = elemMul(B+C, D);

  IndexVar i,j;
  tA(i,j) = (tB(i,j) + tC(i,j)) * tD(i,j);

  ASSERT_TENSOR_EQ(A,tA);
}
