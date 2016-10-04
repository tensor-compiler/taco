#include "test_tensors.h"

namespace taco {
namespace test {

TensorData vector1a({1}, {
  {{0}, 1}
});

TensorData vector5a({5}, {
  {{4}, 2},
  {{1}, 1}
});

TensorData matrix33a({3,3}, {
  {{0,1}, 1},
  {{2,0}, 2},
  {{2,2}, 3}
});

TensorData tensor233a({2,3,3}, {
  {{0,0,0}, 1},
  {{0,0,1}, 2},
  {{0,2,2}, 3},
  {{1,0,1}, 4},
  {{1,2,0}, 5},
  {{1,2,2}, 6}
});

}}
