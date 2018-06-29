#include "taco/storage/array_util.h"

namespace taco {

Array makeArray(DataType type, size_t size) {
  return Array(type, malloc(size * type.getNumBytes()), size, Array::Free);
}

}
