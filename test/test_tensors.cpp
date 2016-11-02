#include <algorithm>

#include "test_tensors.h"

namespace taco {
namespace test {

std::vector<std::vector<LevelType>> generateLevels(size_t order) {
  iassert(order > 0);
  std::vector<size_t> divisors(order);

  const size_t numLevelTypes = 2;

  divisors[0] = 1;
  for (size_t i = 1; i < order; ++i) {
    divisors[i] = numLevelTypes * divisors[i - 1];
  }
  
  const size_t numPermutations = numLevelTypes * divisors[order - 1];

  std::vector<std::vector<LevelType>> levels(numPermutations);
  for (size_t i = 0; i < levels.size(); ++i) {
    std::vector<LevelType> level(order);
    for (size_t j = 0; j < order; ++j) {
      switch ((i / divisors[j]) % numLevelTypes) {
        case 0:
          level[j] = Dense;
          break;
        case 1:
          level[j] = Sparse;
          break;
        //case 2:
        //  level[j] = Fixed;
        //  break;
        default:
          not_supported_yet;
          break;
      }
    }
    levels[i] = level;
  }

  return levels;
}

std::vector<std::vector<size_t>> generateDimensionOrders(size_t order) {
  std::vector<size_t> dimOrder(order);
  for (size_t i = 0; i < order; ++i) {
    dimOrder[i] = i;
  }

  std::vector<std::vector<size_t>> dimOrders;
  do {
    dimOrders.push_back(dimOrder);
  } while (std::next_permutation(dimOrder.begin(), dimOrder.end()));

  return dimOrders;
}

TensorData<double> d1a_data() {
  return TensorData<double>({1}, {
    {{0}, 2}
  });
}

TensorData<double> d1b_data() {
  return TensorData<double>({1}, {
    {{0}, 10}
  });
}

TensorData<double> d5a_data() {
  return TensorData<double>({5}, {
    {{4}, 3},
    {{1}, 2}
  });
}

TensorData<double> d5b_data() {
  return TensorData<double>({5}, {
    {{0}, 10},
    {{1}, 20}
  });
}

TensorData<double> d5c_data() {
  return TensorData<double>({5}, {
    {{1}, 100},
    {{3}, 200},
    {{4}, 300}
  });
}

TensorData<double> d33a_data() {
  return TensorData<double>({3,3}, {
    {{0,1}, 2},
    {{2,0}, 3},
    {{2,2}, 4}
  });
}

TensorData<double> d33b_data() {
  return TensorData<double>({3,3}, {
    {{0,0}, 10},
    {{0,1}, 20},
    {{2,1}, 30}
  });
}

TensorData<double> d233a_data() {
  return TensorData<double>({2,3,3}, {
    {{0,0,0}, 2},
    {{0,0,1}, 3},
    {{0,2,2}, 4},
    {{1,0,1}, 5},
    {{1,2,0}, 6},
    {{1,2,2}, 7}
  });
}

TensorData<double> d233b_data() {
  return TensorData<double>({2,3,3}, {
    {{0,1,0}, 10},
    {{0,1,2}, 20},
    {{0,2,1}, 30},
    {{1,0,2}, 40},
    {{1,2,0}, 50},
    {{1,2,1}, 60}
  });
}

Tensor<double> d1a(std::string name, Format format) {
  return d1a_data().makeTensor(name, format);
}

Tensor<double> d1b(std::string name, Format format) {
  return d1b_data().makeTensor(name, format);
}

Tensor<double> d5a(std::string name, Format format) {
  return d5a_data().makeTensor(name, format);
}

Tensor<double> d5b(std::string name, Format format) {
  return d5b_data().makeTensor(name, format);
}

Tensor<double> d5c(std::string name, Format format) {
  return d5c_data().makeTensor(name, format);
}

Tensor<double> d33a(std::string name, Format format) {
  return d33a_data().makeTensor(name, format);
}

Tensor<double> d33b(std::string name, Format format) {
  return d33b_data().makeTensor(name, format);
}

Tensor<double> d233a(std::string name, Format format) {
  return d233a_data().makeTensor(name, format);
}

Tensor<double> d233b(std::string name, Format format) {
  return d233b_data().makeTensor(name, format);
}

}}
