#include "test.h"
#include "test_tensors.h"
#include "taco/tensor.h"
#include "taco/index_notation/index_notation.h"

using namespace taco;
const IndexVar i("i"), j("j"), k("k");

TEST(concrete, construction) {
  Type t(type<double>(), {3});
  TensorVar a("a", t), b("b", t), c("c", t);
  IndexVar i("i");

  Assignment assignment = (a(i) = b(i) + c(i));
}
