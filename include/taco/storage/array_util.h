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
/// mode index will free/delete the memory or leave the responsibility for
/// freeing to the user.
template <typename T>
Array makeArray(T* data, size_t size, Array::Policy policy=Array::UserOwns) {
  return Array(type<T>(), data, size, policy);
}

/// Construct an array of elements of the given type.
Array makeArray(Type type, size_t size);

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

/// Returns the ith array element as a value of type T. The array type must be
/// compatible with T (compatible type kinds and smaller bit width).
template <typename T> T getValue(const Array& array, size_t i) {
  taco_iassert(i < array.getSize()) << "array index out of bounds";

  // check type compatability
  Type from = array.getType();
  Type to = type<T>();

  if (from == to) {
    return ((T*)array.getData())[i];
  }

  // It's fine to convert a type to a larger type of the same kind
  if (from.getKind() == to.getKind() && from.getNumBits() <= to.getNumBits()) {
    switch (from.getKind()) {
      case Type::Bool:
        break;
      case Type::UInt:
        switch (from.getNumBits()) {
          case 8:
            return (T)(((uint8_t*)array.getData())[i]);
          case 16:
            return (T)(((uint16_t*)array.getData())[i]);
          case 32:
            return (T)(((uint32_t*)array.getData())[i]);
          case 64:
            return (T)(((uint64_t*)array.getData())[i]);
        }
        break;
      case Type::Int:
        switch (from.getNumBits()) {
          case 8:
            return (T)(((int8_t*)array.getData())[i]);
          case 16:
            return (T)(((int16_t*)array.getData())[i]);
          case 32:
            return (T)(((int32_t*)array.getData())[i]);
          case 64:
            return (T)(((int64_t*)array.getData())[i]);
        }
        break;
      case Type::Float:
        switch (from.getNumBits()) {
          case 32:
            return (T)(((float*)array.getData())[i]);
          case 64:
            return (T)(((double*)array.getData())[i]);
        }
        break;
      case Type::Undefined:
        taco_ierror;
        break;
    }
  }

  // Convert (non-negative) integers to unsigned integers
  if (from.getKind() == Type::Int && to.getKind() == Type::UInt &&
      from.getNumBits() <= to.getNumBits()) {
    switch (from.getNumBits()) {
      case 8:
        taco_iassert(((int8_t*)array.getData())[i] >= 0);
        return (T)(((int8_t*)array.getData())[i]);
      case 16:
        taco_iassert(((int16_t*)array.getData())[i] >= 0);
        return (T)(((int16_t*)array.getData())[i]);
      case 32: {
        taco_iassert(((int32_t*)array.getData())[i] >= 0);
        return (T)(((int32_t*)array.getData())[i]);
      }
      case 64:
        taco_iassert(((int64_t*)array.getData())[i] >= 0);
        return (T)(((int64_t*)array.getData())[i]);
    }
  }

  taco_ierror << "Incompatible types " << from << " and " << to;
  return 0;
}

}}
#endif
