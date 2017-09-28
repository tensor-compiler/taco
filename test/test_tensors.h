#ifndef TACO_TEST_TENSORS_H
#define TACO_TEST_TENSORS_H

#include <set>
#include <vector>
#include <utility>
#include <algorithm>

#include "test.h"

#include "taco/tensor.h"
#include "taco/format.h"
#include "taco/util/collections.h"

namespace taco {
namespace test {

std::vector<std::vector<ModeType>> generateModeTypes(size_t order);
std::vector<std::vector<size_t>> generateModeOrderings(size_t order);

template <typename T>
struct TensorData {
  TensorData() = default;
  TensorData(const std::vector<int>& dimensions,
             const vector<std::pair<std::vector<int>,T>>& values) :
      dimensions(dimensions), values(values) {}
  TensorData(const std::vector<int>& dimensions) :
      dimensions(dimensions) {}
  
  Tensor<T> makeTensor(const std::string& name, Format format) const {
    Tensor<T> t(name, dimensions, format);
    for (auto& value : values) {
      t.insert(value.first, value.second);
    }
    return t;
  }

  bool compare(const Tensor<T>&tensor) const {
    if (tensor.getDimensions() != dimensions) {
      return false;
    }

    {
      std::set<std::vector<int>> coords;
      for (const auto& val : tensor) {
        if (!coords.insert(val.first).second) {
          return false;
        }
      }
    }

    vector<std::pair<std::vector<int>,T>> vals;
    for (const auto& val : tensor) {
      if (val.second != 0) {
        vals.push_back(val);
      }
    }

    vector<std::pair<std::vector<int>,T>> expected = this->values;
    std::sort(expected.begin(), expected.end());
    std::sort(vals.begin(), vals.end());
    return vals == expected;
  }

  std::vector<int>                           dimensions;
  std::vector<std::pair<std::vector<int>,T>> values;
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const TensorData<T>&) {
  return os << "TensorData";
}

TensorData<double> da_data();
TensorData<double> db_data();

TensorData<double> d1a_data();
TensorData<double> d1b_data();

TensorData<double> d3b_data();
TensorData<double> d3a_data();

TensorData<double> d4a_data();
TensorData<double> d4b_data();

TensorData<double> d5a_data();
TensorData<double> d5b_data();
TensorData<double> d5c_data();
TensorData<double> d5d_data();

TensorData<double> d8a_data();
TensorData<double> d8b_data();
TensorData<double> d8c_data();

TensorData<double> dla_data();
TensorData<double> dlb_data();

TensorData<double> d33a_data();
TensorData<double> d33at_data();
TensorData<double> d33b_data();
TensorData<double> d33c_data();

TensorData<double> d34a_data();
TensorData<double> d34b_data();

TensorData<double> d44a_data();

TensorData<double> dlla_data();

TensorData<double> d233a_data();
TensorData<double> d233b_data();
TensorData<double> d233c_data();

TensorData<double> d333a_data();

TensorData<double> d32b_data();
TensorData<double> d3322a_data();

Tensor<double> da(std::string name, Format format);
Tensor<double> db(std::string name, Format format);

Tensor<double> d1a(std::string name, Format format);
Tensor<double> d1b(std::string name, Format format);

Tensor<double> d3a(std::string name, Format format);
Tensor<double> d3b(std::string name, Format format);

Tensor<double> d4a(std::string name, Format format);
Tensor<double> d4b(std::string name, Format format);

Tensor<double> d5a(std::string name, Format format);
Tensor<double> d5b(std::string name, Format format);
Tensor<double> d5c(std::string name, Format format);
Tensor<double> d5d(std::string name, Format format);

Tensor<double> d8a(std::string name, Format format);
Tensor<double> d8b(std::string name, Format format);
Tensor<double> d8c(std::string name, Format format);

Tensor<double> dla(std::string name, Format format);
Tensor<double> dlb(std::string name, Format format);

Tensor<double> d33a(std::string name, Format format);
Tensor<double> d33at(std::string name, Format format);
Tensor<double> d33b(std::string name, Format format);
Tensor<double> d33c(std::string name, Format format);

Tensor<double> d34a(std::string name, Format format);
Tensor<double> d34b(std::string name, Format format);

Tensor<double> d44a(std::string name, Format format);

Tensor<double> d55a(std::string name, Format format);
Tensor<double> d35a(std::string name, Format format);
Tensor<double> d53a(std::string name, Format format);

Tensor<double> d233a(std::string name, Format format);
Tensor<double> d233b(std::string name, Format format);
Tensor<double> d233c(std::string name, Format format);

Tensor<double> d333a(std::string name, Format format);

Tensor<double> d32b(std::string name, Format format);
Tensor<double> d3322a(std::string name, Format format);

Tensor<double> d33a_CSR(std::string name);
Tensor<double> d33a_CSC(std::string name);
Tensor<double> d35a_CSR(std::string name);
Tensor<double> d35a_CSC(std::string name);

TensorBase readTestTensor(std::string filename, Format format=Sparse);

}}
#endif
