#ifndef TACO_TEST_TENSORS_H
#define TACO_TEST_TENSORS_H

#include <set>
#include <vector>
#include <utility>
#include <algorithm>

#include "tensor.h"
#include "format.h"

namespace taco {
namespace test {

std::vector<std::vector<LevelType>> generateLevels(size_t order);
std::vector<std::vector<size_t>>    generateDimensionOrders(size_t order);

template <typename T>
struct TensorData {
  typedef typename Tensor<T>::Dimensions      Dimensions;
  typedef std::set<typename Tensor<T>::Value> Values;

  TensorData() = default;
  TensorData(Dimensions dimensions, Values values) : 
      dimensions(dimensions), values(values) {}
  
  Tensor<T> makeTensor(const std::string& name, Format format) const {
    Tensor<T> t(name, dimensions, format);
    t.insert(values.begin(), values.end());
    t.pack();
    return t;
  }

  bool compare(const Tensor<T>&tensor) const {
    {
      std::set<typename Tensor<T>::Coordinate> coords;
      for (const auto& val : tensor) {
        if (!coords.insert(val.first).second) {
          return false;
        }
      }
    }

    Values vals;
    for (const auto& val : tensor) {
      if (val.second != 0) {
        vals.insert(val);
      }
    }

    return vals == values;
  }

  Dimensions dimensions;
  Values     values;
};

TensorData<double> d1a_data();
TensorData<double> d1b_data();

TensorData<double> d5a_data();
TensorData<double> d5b_data();
TensorData<double> d5c_data();

TensorData<double> d33a_data();
TensorData<double> d33b_data();

TensorData<double> d233a_data();
TensorData<double> d233b_data();

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
