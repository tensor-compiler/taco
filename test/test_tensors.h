#ifndef TACO_TEST_TENSORS_H
#define TACO_TEST_TENSORS_H

#include <vector>
#include <utility>

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

extern TensorData vector1a;
extern TensorData vector5a;

extern TensorData matrix33a;

extern TensorData tensor233a;

}}
#endif
