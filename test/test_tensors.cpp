#include <algorithm>

#include "test_tensors.h"

namespace taco {
namespace test {

std::vector<std::vector<ModeFormatPack>> generateModeTypes(int order) {
  taco_iassert(order > 0);
  std::vector<size_t> divisors(order);

  const size_t numModeTypes = 2;

  divisors[0] = 1;
  for (int i = 1; i < order; ++i) {
    divisors[i] = numModeTypes * divisors[i - 1];
  }
  
  const size_t numPermutations = numModeTypes * divisors[order - 1];

  std::vector<std::vector<ModeFormatPack>> levels(numPermutations);
  for (size_t i = 0; i < levels.size(); ++i) {
    std::vector<ModeFormatPack> level;
    for (int j = 0; j < order; ++j) {
      switch ((i / divisors[j]) % numModeTypes) {
        case 0:
          level.push_back(Dense);
          break;
        case 1:
          level.push_back(Sparse);
          break;
        default:
          taco_not_supported_yet;
          break;
      }
    }
    levels[i] = level;
  }

  return levels;
}

std::vector<std::vector<int>> generateModeOrderings(int order) {
  std::vector<int> modeOrdering(order);
  for (int i = 0; i < order; ++i) {
    modeOrdering[i] = i;
  }

  std::vector<std::vector<int>> modeOrderings;
  do {
    modeOrderings.push_back(modeOrdering);
  } while (std::next_permutation(modeOrdering.begin(), modeOrdering.end()));

  return modeOrderings;
}

TensorData<double> da_data() {
  return TensorData<double>({}, {
    {{}, 2}
  });
}

TensorData<double> db_data() {
  return TensorData<double>({}, {
    {{}, 10}
  });
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

TensorData<double> d3a_data() {
  return TensorData<double>({3}, {
    {{0}, 3},
    {{1}, 2},
    {{2}, 1}
  });
}

TensorData<double> d3b_data() {
  return TensorData<double>({3}, {
    {{0}, 2},
    {{2}, 3}
  });
}

TensorData<double> d4a_data() {
  return TensorData<double>({4}, {
    {{0}, 10},
    {{1}, 20}
  });
}

TensorData<double> d4b_data() {
  return TensorData<double>({4}, {
    {{0}, 10},
    {{2}, 20}
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

TensorData<double> d5d_data() {
  return TensorData<double>({5}, {
    {{0}, 1000},
    {{3}, 2000},
    {{4}, 3000}
  });
}

TensorData<double> d5e_data() {
  return TensorData<double>({5}, {
    {{0}, 1},
    {{1}, 2},
    {{2}, 3},
    {{3}, 4},
    {{4}, 5}
  });
}

TensorData<double> d8a_data() {
  return TensorData<double>({8}, {
    {{0}, 1},
    {{1}, 2},
    {{2}, 3},
    {{5}, 4}
  });
}

TensorData<double> d8b_data() {
  return TensorData<double>({8}, {
    {{0}, 10},
    {{2}, 20},
    {{3}, 30}
  });
}

TensorData<double> d8c_data() {
  return TensorData<double>({8}, {
    {{1}, 100},
    {{3}, 200},
    {{5}, 300},
    {{7}, 400}
  });
}

TensorData<double> d8d_data() {
  return TensorData<double>({8}, {
    {{1}, 100},
    {{2}, 200},
    {{5}, 300},
    {{6}, 400}
  });
}

TensorData<double> dla_data() {
  std::vector<std::pair<std::vector<int>,double>> valsList;
  for (int i = 0; i < 10000; ++i) {
    if (i % 2 == 0) {
      valsList.push_back({{i}, (double)i});
    }
  }
  return TensorData<double>({10000}, valsList);
}

TensorData<double> dlb_data() {
  std::vector<std::pair<std::vector<int>,double>> valsList;
  for (int i = 0; i < 10000; ++i) {
    if (i % 3 == 0) {
      valsList.push_back({{i}, (double)i});
    }
  }
  return TensorData<double>({10000}, valsList);
}

TensorData<double> d33a_data() {
  return TensorData<double>({3,3}, {
    {{0,1}, 2},
    {{2,0}, 3},
    {{2,2}, 4}
  });
}

TensorData<double> d33at_data() {
  return TensorData<double>({3,3}, {
    {{1,0}, 2},
    {{0,2}, 3},
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

TensorData<double> d33c_data() {
  return TensorData<double>({3,3}, {
    {{0,1}, 10},
    {{2,0}, 20},
    {{2,1}, 30}
  });
}

TensorData<double> d34a_data() {
  return TensorData<double>({3,4}, {
    {{0,0}, 2},
    {{0,2}, 3},
    {{2,0}, 4},
    {{2,3}, 5}
  });
}

TensorData<double> d34b_data() {
  return TensorData<double>({3,4}, {
    {{0,0}, 2},
    {{0,3}, 3},
    {{2,0}, 4},
    {{2,2}, 5}
  });
}

TensorData<double> d44a_data() {
  return TensorData<double>({4,4}, {
    {{0,0}, 1},
    {{0,2}, 2},
    {{0,3}, 3},
    {{1,1}, 4},
    {{2,2}, 5},
    {{3,1}, 6}
  });
}

TensorData<double> d55a_data() {
  return TensorData<double>({5,5}, {
    {{0,0}, 2},
    {{1,0}, 3},
    {{3,0}, 4},
    {{2,2}, 5},
    {{4,2}, 6},
    {{1,4}, 7},
    {{4,4}, 8}
  });
}

TensorData<double> d35a_data() {
  return TensorData<double>({3,5}, {
    {{0,0}, 2},
    {{2,0}, 3},
    {{0,1}, 4},
    {{2,3}, 5}
  });
}

TensorData<double> d53a_data() {
  return TensorData<double>({5,3}, {
    {{0,0}, 2},
    {{0,2}, 3},
    {{1,0}, 4},
    {{3,2}, 5}
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

TensorData<double> d233c_data() {
  return TensorData<double>({2,3,3}, {
    {{0,0,0}, 100},
    {{0,1,2}, 200},
    {{0,2,1}, 300},
    {{1,2,0}, 400},
    {{1,2,2}, 500}
  });
}

TensorData<double> d333a_data() {
  return TensorData<double>({3,3,3}, {
    {{0,0,0}, 2},
    {{0,0,1}, 3},
    {{0,2,2}, 4},
    {{1,0,1}, 5},
    {{1,2,0}, 6},
    {{1,2,2}, 7},
    {{2,1,2}, 8},
    {{2,2,1}, 9},
  });
}

TensorData<double> d355a_data() {
  return TensorData<double>({3,5,5}, {
    {{0,0,0}, 1},
    {{0,1,1}, 2},
    {{0,2,1}, 3},
    {{0,3,1}, 4},
    {{0,4,1}, 5},
    {{1,0,1}, 6},
    {{1,1,0}, 7},
    {{1,2,0}, 8},
    {{1,4,2}, 9},
    {{2,1,2}, 10},
    {{2,2,3}, 11},
    {{2,4,4}, 12},
  });
}

TensorData<double> d32b_data() {
  return TensorData<double>({3,2}, {
    {{0,0}, 10},
    {{0,1}, 11},
    {{1,0}, 20},
    {{1,1}, 21},
    {{2,0}, 30},
    {{2,1}, 31},
  });
}

TensorData<double> d3322a_data() {
  return TensorData<double>({3,3,2,2}, {
    {{0,1, 0,0}, 2.1},
    {{0,1, 0,1}, 2.2},
    {{0,1, 1,0}, 2.3},
    {{0,1, 1,1}, 2.4},

    {{2,0, 0,0}, 3.1},
    {{2,0, 0,1}, 3.2},
    {{2,0, 1,0}, 3.3},
    {{2,0, 1,1}, 3.4},

    {{2,2, 0,0}, 4.1},
    {{2,2, 0,1}, 4.2},
    {{2,2, 1,0}, 4.3},
    {{2,2, 1,1}, 4.4},
  });
}

Tensor<double> da(std::string name, Format format) {
  return da_data().makeTensor(name, format);
}

Tensor<double> db(std::string name, Format format) {
  return db_data().makeTensor(name, format);
}

Tensor<double> d1a(std::string name, Format format) {
  return d1a_data().makeTensor(name, format);
}

Tensor<double> d1b(std::string name, Format format) {
  return d1b_data().makeTensor(name, format);
}

Tensor<double> d3a(std::string name, Format format) {
  return d3a_data().makeTensor(name, format);
}

Tensor<double> d3b(std::string name, Format format) {
  return d3b_data().makeTensor(name, format);
}

Tensor<double> d4a(std::string name, Format format) {
  return d4a_data().makeTensor(name, format);
}

Tensor<double> d4b(std::string name, Format format) {
  return d4b_data().makeTensor(name, format);
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

Tensor<double> d5d(std::string name, Format format) {
  return d5d_data().makeTensor(name, format);
}

Tensor<double> d5e(std::string name, Format format) {
  return d5e_data().makeTensor(name, format);
}

Tensor<double> d8a(std::string name, Format format) {
  return d8a_data().makeTensor(name, format);
}

Tensor<double> d8b(std::string name, Format format) {
  return d8b_data().makeTensor(name, format);
}

Tensor<double> d8c(std::string name, Format format) {
  return d8c_data().makeTensor(name, format);
}

Tensor<double> d8d(std::string name, Format format) {
  return d8d_data().makeTensor(name, format);
}

Tensor<double> dla(std::string name, Format format) {
  return dla_data().makeTensor(name, format);
}

Tensor<double> dlb(std::string name, Format format) {
  return dlb_data().makeTensor(name, format);
}

Tensor<double> d33a(std::string name, Format format) {
  return d33a_data().makeTensor(name, format);
}

Tensor<double> d33at(std::string name, Format format) {
  return d33at_data().makeTensor(name, format);
}

Tensor<double> d33b(std::string name, Format format) {
  return d33b_data().makeTensor(name, format);
}

Tensor<double> d33c(std::string name, Format format) {
  return d33c_data().makeTensor(name, format);
}

Tensor<double> d34a(std::string name, Format format) {
  return d34a_data().makeTensor(name, format);
}

Tensor<double> d34b(std::string name, Format format) {
  return d34b_data().makeTensor(name, format);
}

Tensor<double> d44a(std::string name, Format format) {
  return d44a_data().makeTensor(name, format);
}

Tensor<double> d55a(std::string name, Format format) {
  return d55a_data().makeTensor(name, format);
}

Tensor<double> d35a(std::string name, Format format) {
  return d35a_data().makeTensor(name, format);
}

Tensor<double> d53a(std::string name, Format format) {
  return d53a_data().makeTensor(name, format);
}

Tensor<double> d233a(std::string name, Format format) {
  return d233a_data().makeTensor(name, format);
}

Tensor<double> d233b(std::string name, Format format) {
  return d233b_data().makeTensor(name, format);
}

Tensor<double> d233c(std::string name, Format format) {
  return d233c_data().makeTensor(name, format);
}

Tensor<double> d333a(std::string name, Format format) {
  return d333a_data().makeTensor(name, format);
}

Tensor<double> d355a(std::string name, Format format) {
  return d355a_data().makeTensor(name, format);
}

Tensor<double> d32b(std::string name, Format format) {
  return d32b_data().makeTensor(name, format);
}

Tensor<double> d3322a(std::string name, Format format) {
  return d3322a_data().makeTensor(name, format);
}


Tensor<double> d33a_CSR(std::string name) {
  return makeCSR<double>(name, {3,3},
                 {0, 1, 1, 3},
                 {1, 0, 2},
                 {2.0,3.0,4.0});
}

Tensor<double> d33a_CSC(std::string name) {
  return makeCSR<double>(name, {3,3},
					  {0, 1, 2, 3},
					  {2, 0, 2},
                 {3.0,2.0,4.0});
}

Tensor<double> d35a_CSR(std::string name) {
  return makeCSR<double>(name, {3,5},
					  {0, 2, 2, 4},
					  {0, 1, 0, 3},
                 {2.0,4.0,3.0,5.0});
}

Tensor<double> d35a_CSC(std::string name) {
  return makeCSC<double>(name, {3,5},
					  {0, 2, 3, 3, 4, 4},
					  {0, 2, 0, 2},
                 {2.0,3.0,4.0,5.0});
}

Tensor<double> d33a(std::string name, ModeFormat modeType) {
  return d33a_data().makeTensor(name, modeType);
}

TensorBase readTestTensor(std::string filename, Format format) {
  return read(testDirectory()+"/data/"+filename, format, false);
}

}}
