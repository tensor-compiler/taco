#include "packed_tensor.h"

#include <iostream>
#include <string>

#include "util/strings.h"

using namespace std;

namespace taco {

std::ostream& operator<<(std::ostream& os, const PackedTensor& tp) {
  auto& levelStorage = tp.getLevelStorage();
  double* values = tp.getValues();
  auto nnz       = tp.getNnz();

  // Print indices
  for (size_t i=0; i < levelStorage.size(); ++i) {
    auto& level = levelStorage[i];
    os << "L" << to_string(i) << ":" << std::endl;
    os << "  idx: {" << util::join(level.ptr) << "}" << std::endl;
    os << "  ptr: {" << util::join(level.idx) << "}" << std::endl;
  }

  //  // Print values
  os << "vals:  {" << util::join(values, values+nnz) << "}";

  return os;
}

}
