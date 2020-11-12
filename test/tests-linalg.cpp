#include "test.h"

#include "taco/linalg.h"

using namespace taco;

TEST(linalg, matmul) {
  Matrix<double> B("B", 2, 2, dense, dense);
  Matrix<double> C("C", 2, 2, dense, dense);
  Matrix<double> A("A", 2, 2, dense, dense);

  cout << "--- Before inserting ---" << endl;
  B.insert(0,0,2);
  B.insert(1,1,1);
  B.insert(0,1,2);

  C.insert(0,0,2);
  C.insert(1,1,2);
  cout << "--- After inserting ---" << endl;

  cout << "--- Before Expression ---" << endl;
  A = B * C;
  cout << "--- After Expression ---" << endl;

  cout << "--- Before At ---" << endl;
  cout << "B(0,0): " << B.at(0,0) << endl;
  cout << "A(0,0): " << A.at(0,0) << endl;
  cout << "--- After At ---" << endl;

  cout << "--- Before Rewrite of A ---" << endl;
  A.rewrite();
  cout << "--- After Rewrite of A ---" << endl;

  cout << "--- Before At (A) ---" << endl;
  cout << "A(0,0): " << A.at(0,0) << endl;
  cout << "--- After At (A) ---" << endl;

  cout << "--- before cout of a ---" << endl;
  cout << A << endl;
  cout << "--- after cout of a ---" << endl;

  cout << "--- Before getIndexAssignment on A ---" << endl;
  cout << A.getIndexAssignment() << endl;
  cout << "--- After getIndexAssignment on A ---" << endl;
}

TEST(linalg, tensorbase) {
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
