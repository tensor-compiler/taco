#ifndef TACO_DISTRIBUTION_H
#define TACO_DISTRIBUTION_H

#include <vector>

namespace taco {

// Grid represents an n-dimensional grid.
class Grid {
public:
  Grid() {};

  // Variadic template style constructor for grids.
  template <typename... Args>
  Grid(Args... args) : Grid(std::vector<int>{args...}) {}

  Grid(std::vector<int>& dims) {
    this->dimensions = dims;
  }

  Grid(std::vector<int>&& dims) {
    this->dimensions = dims;
  }

  int getDim() {
    return this->dimensions.size();
  }

  int getDimSize(int dim) {
    return this->dimensions[dim];
  }

private:
  std::vector<int> dimensions;
};

}

#endif //TACO_DISTRIBUTION_H
