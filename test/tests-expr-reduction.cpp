#include "test.h"
#include "test_tensors.h"
#include "taco/tensor.h"

using namespace taco;

const IndexVar i("i"), j("j"), k("k");

TEST(expr, reduction0) {
  Tensor<double> a("a");

  Tensor<double> b = d5a("b", Dense);
  Tensor<double> c = d5b("c", Dense);
  b.pack();
  c.pack();

  a = sum(i, b(i)*c(i));
  a.evaluate();

  Tensor<double> expected("expected");
  expected.insert({}, 40.0);
  expected.pack();
  ASSERT_TRUE(equals(expected,a)) << endl << expected << endl << endl << a;;
}
