#ifndef TACO_TYPE_H
#define TACO_TYPE_H

#include <ostream>
#include <cstdint>
#include "taco/error.h"

namespace taco {

/// A basic taco type. These can be boolean, integer, unsigned integer, float
/// or complex float at different precisions.
class DataType {
public:
  /// The kind of type this object represents.
  enum Kind {
    Bool,

    /// Unsigned integer (8, 16, 32, 64)
    UInt,

    /// Signed integer (8, 16, 32, 64)
    Int,

    /// Floating point (32, 64)
    Float,

    /// Undefined type
    Undefined
  };

  /// Construct an undefined type.
  DataType();

  /// Construct a taco basic type with default bit widths.
  DataType(Kind);

  /// Construct a taco basic type with the given bit width.
  DataType(Kind, size_t bits);

  /// Return the kind of type this object represents.
  Kind getKind() const;

  /// Functions that return true if the type is the given type.
  /// @{
  bool isBool() const;
  bool isUInt() const;
  bool isInt() const;
  bool isFloat() const;
  /// @}

  /// Returns the number of bytes required to store one element of this type.
  size_t getNumBytes() const;

  /// Returns the number of bits required to store one element of this type.
  size_t getNumBits() const;

private:
  Kind kind;
  size_t bits;
};

std::ostream& operator<<(std::ostream&, const DataType&);
std::ostream& operator<<(std::ostream&, const DataType::Kind&);
bool operator==(const DataType& a, const DataType& b);
bool operator!=(const DataType& a, const DataType& b);

/// Construct a float with the given bit width
DataType Bool(size_t bits = sizeof(bool));
DataType Int(size_t bits);
DataType UInt(size_t bits);
DataType Float(size_t bits);

template <typename T>
typename std::enable_if<std::is_integral<T>::value &&
                       std::is_signed<T>::value, DataType>::type type() {
  return DataType(DataType::Int, sizeof(T)*8);
}

template <typename T>
typename std::enable_if<std::is_integral<T>::value &&
                       !std::is_signed<T>::value, DataType>::type type() {
  return DataType(DataType::UInt, sizeof(T)*8);
}

template <typename T>
typename std::enable_if<std::is_floating_point<T>::value,DataType>::type type(){
  return DataType(DataType::Float, sizeof(T)*8);
}

}
#endif
