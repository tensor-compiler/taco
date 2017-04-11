#ifndef TACO_TEST_TENSORS_H
#define TACO_TEST_TENSORS_H

#include <set>
#include <vector>
#include <utility>
#include <algorithm>

#include "test.h"

#include "taco/tensor.h"
#include "taco/format.h"

namespace taco {
namespace test {

std::vector<std::vector<LevelType>> generateLevels(size_t order);
std::vector<std::vector<size_t>>    generateDimensionOrders(size_t order);
const Format CSR({Dense, Sparse},{0,1});
const Format CSC({Dense, Sparse},{1,0});

template <typename T>
struct TensorData {
  typedef std::vector<typename Tensor<T>::Value> Values;

  TensorData() = default;
  TensorData(const std::vector<int>& dimensions, const Values& values) :
      dimensions(dimensions), values(values) {}
  TensorData(const std::vector<int>& dimensions) :
      dimensions(dimensions) {}
  
  Tensor<T> makeTensor(const std::string& name, Format format) const {
    Tensor<T> t(name, dimensions, format);
    t.insert(values.begin(), values.end());
    t.pack();
    return t;
  }

  Tensor<T> setCSR(const std::string& name, std::vector<T> A,
                   std::vector<int> IA, std::vector<int> JA) const {
    Tensor<T> t(name, dimensions, CSR);
    t.setCSR(util::copyToArray(A),util::copyToArray(IA),util::copyToArray(JA));
    return t;
  }

  Tensor<T> setCSC(const std::string& name, std::vector<T> val,
                   std::vector<int> row_ind, std::vector<int> col_ptr) const {
    Tensor<T> t(name, dimensions, CSC);
    t.setCSC(util::copyToArray(val), util::copyToArray(row_ind),
             util::copyToArray(col_ptr));
    return t;
  }

  Tensor<T> readFile(const std::string& name,
                     const std::string& filename) const {
    std::string datafilename = testDirectory() + "/data/" + filename;
    Tensor<T> t(name, dimensions, CSC);
    t.read(datafilename);
    return t;
  }

  bool compare(const Tensor<T>&tensor) const {
    if (tensor.getDimensions() != dimensions) {
      return false;
    }

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
        vals.push_back(val);
      }
    }

    Values expected = this->values;
    std::sort(expected.begin(), expected.end());
    std::sort(vals.begin(), vals.end());
    return vals == expected;
  }

  std::vector<int> dimensions;
  Values           values;
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

TensorData<double> d8a_data();
TensorData<double> d8b_data();
TensorData<double> d8c_data();

TensorData<double> dla_data();
TensorData<double> dlb_data();

TensorData<double> d33a_data();
TensorData<double> d33at_data();
TensorData<double> d33b_data();
TensorData<double> d33c_data();

TensorData<double> d44a_data();

TensorData<double> dlla_data();

TensorData<double> d233a_data();
TensorData<double> d233b_data();

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

Tensor<double> d8a(std::string name, Format format);
Tensor<double> d8b(std::string name, Format format);
Tensor<double> d8c(std::string name, Format format);

Tensor<double> dla(std::string name, Format format);
Tensor<double> dlb(std::string name, Format format);

Tensor<double> d33a(std::string name, Format format);
Tensor<double> d33at(std::string name, Format format);
Tensor<double> d33b(std::string name, Format format);
Tensor<double> d33c(std::string name, Format format);

Tensor<double> d44a(std::string name, Format format);

Tensor<double> d55a(std::string name, Format format);
Tensor<double> d35a(std::string name, Format format);
Tensor<double> d53a(std::string name, Format format);

Tensor<double> d233a(std::string name, Format format);
Tensor<double> d233b(std::string name, Format format);

Tensor<double> d333a(std::string name, Format format);

Tensor<double> d32b(std::string name, Format format);
Tensor<double> d3322a(std::string name, Format format);

Tensor<double> d33a_CSR(std::string name);
Tensor<double> d33a_CSC(std::string name);
Tensor<double> d35a_CSR(std::string name);
Tensor<double> d35a_CSC(std::string name);
Tensor<double> rua32(std::string name);
Tensor<double> arc130(std::string name);
Tensor<double> arc130mtx(std::string name);
Tensor<double> d33a_MTX(std::string name);

}}
#endif
