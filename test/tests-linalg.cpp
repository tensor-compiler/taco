#include "test.h"

#include "taco/linalg.h"

using namespace taco;

TEST(linalg, simplest) {
  Matrix<double> B("B", 2, 2, dense, dense);
  Matrix<double> C("C", 2, 2, dense, dense);
  Matrix<double> A("A", 2, 2, dense, dense);

  /* Vector<double> c("c"); */

  /* Vector<double> a("a"); */

  /* for(int i=0;i<42;i++) { */
  /*   B.insert({i,i}, 1.0); */
  /* } */

  /* for(int i=0;i<42;i++) { */
  /*   c.insert({i}, (double) i); */
  /* } */

  /* B.pack(); */
  /* c.pack(); */

  /* IndexVar i("i"), j("j"); */

  /* a(i) = B(i,j) * c(j); */

  /* A = B*C; */

  cout << "--- Before Ping ---" << endl;
  B.ping();
  cout << "--- Post-Ping ---" << endl;

  cout << "--- Before Expression ---" << endl;
  A = B * C;
  cout << "--- After Expression ---" << endl;

  cout << "--- Before At ---" << endl;
  cout << "B(0,0): " << B.at(0,0) << endl;
  cout << "--- After At ---" << endl;


  cout << "--- Before cout of A ---" << endl;
  /* cout << A << endl; */
  cout << "--- After cout of A ---" << endl;

  cout << "--- Before Rewrite of A ---" << endl;
  A.rewrite();
  cout << "--- After Rewrite of A ---" << endl;

  cout << "--- Before getIndexAssignment on A ---" << endl;
  cout << A.getIndexAssignment() << endl;
  cout << "--- After getIndexAssignment on A ---" << endl;

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

