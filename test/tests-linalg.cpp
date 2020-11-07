#include "test.h"

#include "taco/linalg.h"

using namespace taco;

TEST(linalg, simplest) {
  Matrix<double> B("B", 2, 2, dense, dense);
  Matrix<double> C("C");
  Matrix<double> A("A");

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

  B.ping();

  A = B * C;

  cout << "A(0,0): " << A.at(0,0) << endl;

  cout << A << endl;

  ASSERT_TRUE(1);
}

