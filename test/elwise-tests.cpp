#include "test.h"
#include "test_tensors.h"

#include "tensor.h"
#include "expr.h"
#include "operator.h"

TEST(elwise, add_vector_to_self_dense) {
  Var i("i");

  // TODO: Should not need to specify dimensions here
  Tensor<double> a({5}, "d");
  auto b = vectord5a("d");
  b.pack();

  a(i) = b(i) + b(i);

  a.compile();

//  a.materialize();
//  std::cout << b << std::endl;
//  std::cout << a << std::endl;
}
