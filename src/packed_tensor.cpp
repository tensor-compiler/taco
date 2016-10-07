#include "packed_tensor.h"

#include <iostream>

#include "util/strings.h"

namespace taco {

std::ostream& operator<<(std::ostream& os, const PackedTensor& tp) {
  std::cout << "-- before" << std::endl;

  auto& indices = tp.getIndices();
  auto& values  = tp.getValues();

  std::cout << "-- after" << std::endl << std::endl;
  // Print indices
  for (size_t i=0; i < indices.size(); ++i) {
    auto& index = indices[i];
    os << "indices:" << std::endl;
    for (size_t j=0; j < index.size(); ++j) {
      auto& indexArray = index[j];
      os << "  {" << util::join(indexArray) << "}" << std::endl;
    }
  }

  //  // Print values
  os << "values:" << std::endl << "  {" << util::join(values) << "}";

  return os;
}

}
