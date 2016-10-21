#ifndef TACO_TEST_TENSORS_H
#define TACO_TEST_TENSORS_H

#include <vector>
#include <utility>

#include "tensor.h"

namespace taco {
namespace test {

struct TestData {
    TestData(Tensor<double> tensor,
                   const PackedTensor::Indices& expectedIndices,
                              const vector<double> expectedValues)
            : tensor(tensor),
                    expectedIndices(expectedIndices), expectedValues(expectedValues) {
                        }
      Tensor<double> tensor;

        // Expected values
        //   PackedTensor::Indices expectedIndices;
        //     vector<double> expectedValues;
        //     };
        //
        //     ostream &operator<<(ostream& os, const TestData& data) {
        //       os << util::join(data.tensor.getDimensions(), "x")
        //            << " (" << data.tensor.getFormat() << ")";
        //              return os;
        //              }

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

Tensor<double> vectord1a(std::string format);
Tensor<double> vectord5a(std::string format);

Tensor<double> matrixd33a(std::string format);

Tensor<double> tensord233a(std::string format);

}}
#endif
