#include "taco/storage/array_util.h"

namespace taco {
namespace storage {

Array makeArray(Type type, size_t size) {
  return Array(type, malloc(size * type.getNumBytes()), size, Array::Free);
}

}}
