#ifndef TACO_STORAGE_ARRAY_UTIL_H
#define TACO_STORAGE_ARRAY_UTIL_H

#include <vector>
#include <initializer_list>

#include "taco/storage/array.h"
#include "taco/type.h"
#include "taco/error.h"

#include "taco/util/collections.h"

namespace taco {
namespace storage {

/// Construct an index array. The ownership policy determines whether the
/// dimension index will free/delete the memory or leave the responsibility for
/// freeing to the user.
template <typename T>
Array makeArray(T* data, size_t size, Array::Policy policy=Array::UserOwns) {
  return Array(type<T>(), data, size, policy);
}

/// Construct an Array from the values.
template <typename T>
Array makeArray(const std::vector<T>& values) {
  return makeArray(util::copyToArray(values), values.size(), Array::Free);
}

/// Construct an Array from the values.
template <typename T>
Array makeArray(const std::initializer_list<T>& values) {
  return makeArray(util::copyToArray(values), values.size(), Array::Free);
}
}}
#endif
