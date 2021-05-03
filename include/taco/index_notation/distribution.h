#ifndef TACO_DISTRIBUTION_H
#define TACO_DISTRIBUTION_H

#include <vector>
#include "taco/index_notation/index_notation.h"

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

  bool defined() { return this->dimensions.size() > 0; }

private:
  std::vector<int> dimensions;
};

// Transfer represents requesting a portion of data.
// TODO (rohany): It seems like we're doing all equality on tensorvars, rather than the access.
//  That is fine, will just need to remember.
class Transfer {
public:
  Transfer() : content(nullptr) {};
  Transfer(taco::Access a);
  Access getAccess() const;
  friend bool operator==(Transfer& a, Transfer&b);
private:
  struct Content;
  std::shared_ptr<Content> content;
};
typedef std::vector<Transfer> Transfers;
std::ostream& operator<<(std::ostream&, const Transfer&);

}

#endif //TACO_DISTRIBUTION_H
