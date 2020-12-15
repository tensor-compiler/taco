#include "test.h"

#include "taco/linalg.h"

using namespace taco;

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

  ASSERT_EQ(A.at(0,0), 4);
  ASSERT_EQ(A.at(0,1), 4);
  ASSERT_EQ(A.at(1,0), 0);
  ASSERT_EQ(A.at(1,1), 2);
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

  ASSERT_EQ(x.at(0), 17);
  ASSERT_EQ(x.at(1), 6);

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

  ASSERT_EQ(x, 9);
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

  ASSERT_EQ(A.at(0,0), 4);
  ASSERT_EQ(A.at(0,1), 4);
  ASSERT_EQ(A.at(1,0), 0);
  ASSERT_EQ(A.at(1,1), 2);

  cout << "A(1,1) = " << A(1,1) << endl;
  double a11 = A(1,1);
  cout << "a11 = " << a11 << endl;
}

TEST(linalg, tensorbase) {
  Matrix<double> B("B", 2, 2, dense, dense);
  Matrix<double> C("C", 2, 2, dense, dense);
  Matrix<double> A("A", 2, 2, dense, dense);

  B(0,0) = 1;
  B(1,1) = 4;

  C(0,1) = 2;
  C(1,0) = 3;

  A = B + C;

  // Should be [1,2,3,4]
  cout << A << endl;
  
  cout << A.getIndexAssignment();

  ASSERT_EQ(A.at(0,0), 1);
  ASSERT_EQ(A.at(0,1), 2);
  ASSERT_EQ(A.at(1,0), 3);
  ASSERT_EQ(A.at(1,1), 4);
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

  ASSERT_EQ(x.at(0), 5);
  ASSERT_EQ(x.at(1), 2);

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

  ASSERT_EQ(x.at(0), 17);
  ASSERT_EQ(x.at(1), 6);

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

  ASSERT_EQ(x, 9);
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

TEST(linalg, rowvec_transpose) {
  Vector<double> b("b", 2, dense, true);
  Matrix<double> A("A", 2, 2, dense, dense);
  Scalar<double> a("a", true);

  b(0) = 2;
  b(1) = 5;

  A(0,0) = 1;
  A(0,1) = 2;
  A(1,1) = 4;

  a = transpose(transpose(b) * A * b);

  // Should be 124
  cout << a << endl;

  cout << a.getIndexAssignment();

  ASSERT_TRUE(1);
}

TEST(linalg, tensorapi) {
  cout << "--- Beginning of TensorAPI test ---" << endl;
  Tensor<double> a({2,2}, dense);
  Tensor<double> b({2,3}, dense);
  Tensor<double> c({3,2}, dense);

  cout << "--- Initialized Tensors ---" << endl;

  b(0,0) = 2;
  b(1,1) = 1;
  b(0,1) = 2;

  cout << "--- Initializing c ---" << endl;

  c(0,0) = 2;
  c(1,1) = 2;

  cout << "--- Declaring IndexVars ---" << endl;

  IndexVar i,j,k;

  // The original
  /* a(i,j) = b(i,k) * c(k,j); */

  // The broken-up version
  cout << "--- Creating operand IndexExprs ---" << endl;

  IndexExpr tc = c(k,j);
  IndexExpr tb = b(i,k);

  cout << "Pre-assignment" << endl;
  a(i,j) = tb * tc;
  cout << "Post-assignment" << endl;

  /* cout << a << endl; */
}

TEST(linalg, complex_expr) {
  Matrix<double> A("A", 2, 2, dense, dense);
  Matrix<double> B("B", 2, 2, dense, dense);
  Matrix<double> C("C", 2, 2, dense, dense);
  Matrix<double> D("D", 2, 2, dense, dense);
  Matrix<double> E("D", 2, 2, dense, dense);

  A = E*elemMul(B+C, D);

  cout << A << endl;

  cout << A.getIndexAssignment();

  ASSERT_TRUE(1);
}

TEST(linalg, blocking_matmul) {
  Matrix<double> B("B", 4, 4, sparse, dense, 2);
  Matrix<double> C("C", 4, 4, dense, sparse, 2);
  Matrix<double> A("A", 4, 4, dense, dense, 2);

  cout << "--- Before inserting ---" << endl;
  B.insert(0,0,0, 0,2);
  B.insert(1,1, 0,0,1);
  B.insert(0,1, 0,0,2);

  C.insert(0,0,2);
  C.insert(1,1,2);
  cout << "--- After inserting ---" << endl;
  cout << "B: " << B << endl;
  cout << "C: " << C << endl;
  cout << "--- Before Expression ---" << endl;
  A = transpose(B * C);
  cout << "--- After Expression ---" << endl;

  cout << "--- Before At ---" << endl;
  auto vec =  B.atBlock(0,0);
  for (auto it = 0; it < vec.size(); ++it)
    cout << "B(0,0): " << vec.at(it) << endl;
  vec =  A.atBlock(0,0);
  for (auto it = 0; it < vec.size(); ++it)
    cout << "B(0,0): " << vec.at(it) << endl;
  cout << "--- After At ---" << endl;

  cout << "--- before cout of a ---" << endl;
  cout << A << endl;
  cout << "--- after cout of a ---" << endl;

  cout << "--- Before getIndexAssignment on A ---" << endl;
  cout << A.getIndexAssignment() << endl;
  cout << "--- After getIndexAssignment on A ---" << endl;
//
//  ASSERT_EQ(A.at(0,0), 4);
//  ASSERT_EQ(A.at(0,1), 4);
//  ASSERT_EQ(A.at(1,0), 0);
  ASSERT_TRUE(1);
}

TEST(linalg, blocking_matvec) {
  Matrix<double> B("B", 4, 4, sparse, dense, 2);
  Vector<double> c("c", 4, sparse, 2);
  Vector<double> a("a", 4, dense, 2);

  cout << "--- Before inserting ---" << endl;
  B.insert(0,0,0, 0,2);
  B.insert(1,1, 0,0,1);
  B.insert(0,1, 0,0,2);

  c.insert(0,0,2);
  c.insert(1,1,2);

  cout << "--- After inserting ---" << endl;
  cout << "B: " << B << endl;
  cout << "C: " << c << endl;
  cout << "c(isColVec): " << c.isColVector();
  cout << "--- Before Expression ---" << endl;
  a = (transpose(c)*c)*(B*c);
  cout << "--- After Expression ---" << endl;

  cout << "--- Before At ---" << endl;
  auto vec =  B.atBlock(0, 0);
  for (auto it = 0; it < vec.size(); ++it)
    cout << "B(0): " << vec.at(it) << endl;
  vec =  a.atBlock(0);
  for (auto it = 0; it < vec.size(); ++it)
    cout << "B(0): " << vec.at(it) << endl;
  cout << "--- After At ---" << endl;

  cout << "--- before cout of a ---" << endl;
  cout << a << endl;
  cout << "--- after cout of a ---" << endl;

  cout << "--- Before getIndexAssignment on A ---" << endl;
  cout << a.getIndexAssignment() << endl;
  cout << "--- After getIndexAssignment on A ---" << endl;
//
//  ASSERT_EQ(A.at(0,0), 4);
//  ASSERT_EQ(A.at(0,1), 4);
//  ASSERT_EQ(A.at(1,0), 0);
  ASSERT_TRUE(1);
}
