#include "test_tensors.h"

namespace taco {
namespace test {

Tensor<double> vectord1a(const std::string& format) {
  return Tensor<double>({1}, format, {
    {{0}, 1}
  });
}

Tensor<double> vectord5a(const std::string& format) {
  return Tensor<double>({5}, format, {
    {{4}, 2},
    {{1}, 1}
  });
}

Tensor<double> matrixd33a(const std::string& format) {
  return Tensor<double>({3,3}, format, {
    {{0,1}, 1},
    {{2,0}, 2},
    {{2,2}, 3}
  });
}

Tensor<double> tensord233a(const std::string& format) {
  return Tensor<double>({2,3,3}, format, {
    {{0,0,0}, 1},
    {{0,0,1}, 2},
    {{0,2,2}, 3},
    {{1,0,1}, 4},
    {{1,2,0}, 5},
    {{1,2,2}, 6}
  });
}

}}
