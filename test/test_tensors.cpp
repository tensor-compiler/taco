#include "test_tensors.h"

namespace taco {
namespace test {

Tensor<double> vectord1a(std::string format) {
  Tensor<double> t({1}, format);
  t.insert({
    {{0}, 1}
  });
  return t;
}

Tensor<double> vectord5a(std::string format) {
  Tensor<double> t({5}, format);
  t.insert({
    {{4}, 2},
    {{1}, 1}
  });
  return t;
}

Tensor<double> matrixd33a(std::string format) {
  Tensor<double> t({3,3}, format);
  t.insert({
    {{0,1}, 1},
    {{2,0}, 2},
    {{2,2}, 3}
  });
  return t;
}

Tensor<double> tensord233a(std::string format) {
  Tensor<double> t({2,3,3}, format);
  t.insert({
    {{0,0,0}, 1},
    {{0,0,1}, 2},
    {{0,2,2}, 3},
    {{1,0,1}, 4},
    {{1,2,0}, 5},
    {{1,2,2}, 6}
  });
  return t;
}

}}
