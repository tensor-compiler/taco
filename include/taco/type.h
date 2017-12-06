#ifndef TACO_TYPE_H
#define TACO_TYPE_H

#include <ostream>
#include <cstdint>
#include <vector>
#include <initializer_list>
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


/// A tensor dimension is the size of a tensor mode.  Tensor dimensions can be
/// variable or fixed sized, which impacts code generation.  Variable dimensions
/// are provided to kernels as arguments, while fixed dimensions are compiled
/// into the kernel.
class Dimension {
public:
  /// Create a variable sized dimension.
  Dimension();

  /// Create a fixed sized dimension.
  Dimension(size_t size);

  /// True if the dimension is variable size, false otherwise.
  bool isVariable() const;

  /// True if the dimension is fixed size, false otherwise.
  bool isFixed() const;

  /// Returns the size of the dimension or 0 if it is variable sized.
  size_t getSize() const;

private:
  size_t size;
};

/// Print a tensor dimension.
std::ostream& operator<<(std::ostream&, const Dimension&);


/// A tensor shape consists of the tensor's dimensions.
class Shape {
public:
  /// Create a default tensor shape: [].
  Shape();

  /// Create a tensor shape.
  Shape(std::initializer_list<Dimension> dimensions);

  /// Create a tensor shape.
  Shape(std::vector<Dimension> dimensions);

  /// Returns the number of dimensions in the shape.
  size_t getOrder() const;

  /// Returns the ith dimension.
  Dimension getDimension(size_t i) const;

  /// Iterator to the first dimension.
  std::vector<Dimension>::const_iterator begin() const;

  /// Iterator past the last dimension.
  std::vector<Dimension>::const_iterator end() const;

private:
  std::vector<Dimension> dimensions;
};

/// Print a tensor shape.
std::ostream& operator<<(std::ostream&, const Shape&);


/// A tensor type consists of a shape and a component/data type.
class Type {
public:
  /// Create a default tensor type (double scalar)
  Type();

  /// Create a tensor type.
  Type(DataType, Shape={});

  DataType getDataType() const;
  Shape getShape() const;

private:
  DataType dtype;
  Shape shape;
};

/// Print a tensor type.
std::ostream& operator<<(std::ostream&, const Type&);

}
#endif
