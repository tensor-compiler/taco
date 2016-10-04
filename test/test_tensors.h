#ifndef TACO_TEST_TENSORS_H
#define TACO_TEST_TENSORS_H

#include <vector>
#include <utility>

#include "tensor.h"

namespace taco {
namespace test {

struct TensorValue {
  TensorValue(std::vector<int> coord, double value)
      : coord(coord), value(value) {}
  std::vector<int> coord;
  double value;
};

typedef std::vector<size_t>      Dimensions;
typedef std::vector<TensorValue> TensorValues;

struct TensorData {
  TensorData(Dimensions dimensions, TensorValues values)
      : dimensions(dimensions), values(values) {}
  Dimensions dimensions;
  TensorValues values;
};

Tensor<double> vectord1a(const std::string& format);
Tensor<double> vectord5a(const std::string& format);

Tensor<double> matrixd33a(const std::string& format);

Tensor<double> tensord233a(const std::string& format);

}}
#endif
