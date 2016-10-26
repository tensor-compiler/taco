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

Tensor<double> d1a(std::string name, Format format);
Tensor<double> d1b(std::string name, Format format);

Tensor<double> d5a(std::string name, Format format);
Tensor<double> d5b(std::string name, Format format);
Tensor<double> d5c(std::string name, Format format);

Tensor<double> d33a(std::string name, Format format);
Tensor<double> d33b(std::string name, Format format);

Tensor<double> d233a(std::string name, Format format);
Tensor<double> d233b(std::string name, Format format);

}}
#endif
