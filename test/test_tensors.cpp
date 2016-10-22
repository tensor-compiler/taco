#include "test_tensors.h"

using namespace std;

namespace taco {
namespace test {

template <typename T>
static Tensor<T> tensor(string name, vector<size_t> dims, Format format,
                        vector<pair<vector<uint32_t>,T>> vals) {
  Tensor<T> t(name, dims, format);
  t.insert(vals);
  t.pack();
  return t;
}

Tensor<double> d1a(std::string name, Format format) {
  return tensor<double>(name, {1}, format, {
    {{0}, 1}
  });
}

Tensor<double> d1b(std::string name, Format format) {
  return tensor<double>(name, {1}, format, {
    {{0}, 10}
  });
}


Tensor<double> d5a(std::string name, Format format) {
  return tensor<double>(name, {5}, format, {
    {{4}, 2},
    {{1}, 1}
  });
}

Tensor<double> d5b(std::string name, Format format) {
  return tensor<double>(name, {5}, format, {
    {{0}, 10},
    {{1}, 20},
  });
}

Tensor<double> d33a(std::string name, Format format) {
  return tensor <double>(name, {3,3}, format, {
    {{0,1}, 1},
    {{2,0}, 2},
    {{2,2}, 3}
  });
}

Tensor<double> d33b(std::string name, Format format) {
  return tensor <double>(name, {3,3}, format, {
    {{0,0}, 10},
    {{0,1}, 20},
    {{2,1}, 30}
  });
}

Tensor<double> d233a(std::string name, Format format) {
  return tensor<double>(name, {2,3,3}, format, {
    {{0,0,0}, 1},
    {{0,0,1}, 2},
    {{0,2,2}, 3},
    {{1,0,1}, 4},
    {{1,2,0}, 5},
    {{1,2,2}, 6}
  });
}

Tensor<double> d233b(std::string name, Format format) {
  return tensor<double>(name, {2,3,3}, format, {
    {{0,0,0}, 10},
    {{0,0,2}, 20},
    {{0,2,1}, 30},
    {{1,0,2}, 40},
    {{1,2,0}, 50},
    {{1,2,1}, 60}
  });
}

}}
