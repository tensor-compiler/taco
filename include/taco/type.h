#ifndef TACO_TYPE_H
#define TACO_TYPE_H

#include <ostream>
#include <cstdint>
#include <vector>
#include <initializer_list>
#include "taco/error.h"
#include <complex>

namespace taco {

/// A basic taco type. These can be boolean, integer, unsigned integer, float
/// or complex float at different precisions.
class DataType {
public:
  /// The kind of type this object represents.
  enum Kind {
    Bool,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Int8,
    Int16,
    Int32,
    Int64,
    Float32,
    Float64,
    Complex64,
    Complex128,

    /// Undefined type
    Undefined
  };

  /// Construct an undefined type.
  DataType();

  /// Construct a taco basic type with default bit widths.
  DataType(Kind);

  /// Return the kind of type this object represents.
  Kind getKind() const;

  /// Functions that return true if the type is the given type.
  /// @{
  bool isUInt() const;
  bool isInt() const;
  bool isFloat() const;
  bool isComplex() const;
  bool isBool() const;
  /// @}

  /// Returns the number of bytes required to store one element of this type.
  size_t getNumBytes() const;

  /// Returns the number of bits required to store one element of this type.
  size_t getNumBits() const;

private:
  Kind kind;
};

std::ostream& operator<<(std::ostream&, const DataType&);
std::ostream& operator<<(std::ostream&, const DataType::Kind&);
bool operator==(const DataType& a, const DataType& b);
bool operator!=(const DataType& a, const DataType& b);

DataType Bool();
DataType UInt8();
DataType UInt16();
DataType UInt32();
DataType UInt64();
DataType Int8();
DataType Int16();
DataType Int32();
DataType Int64();
DataType Float32();
DataType Float64();
DataType Complex64();
DataType Complex128();

  
template<typename T> inline DataType type() {
  taco_ierror << "Unsupported type";
  return DataType(DataType::Int32);
}
  
template<> inline DataType type<bool>() {
  return DataType(DataType::Bool);
}

template<> inline DataType type<unsigned char>() {
  return DataType(DataType::UInt8);
}
  
template<> inline DataType type<unsigned int>() {
  return DataType(DataType::UInt16);
}
  
template<> inline DataType type<unsigned long>() {
  return DataType(DataType::UInt32);
}

template<> inline DataType type<unsigned long long>() {
  return DataType(DataType::UInt64);
}

template<> inline DataType type<char>() {
  return DataType(DataType::Int8);
}

template<> inline DataType type<int>() {
  return DataType(DataType::Int16);
}

template<> inline DataType type<long>() {
  return DataType(DataType::Int32);
}

template<> inline DataType type<long long>() {
  return DataType(DataType::Int64);
}

template<> inline DataType type<float>() {
  return DataType(DataType::Float32);
}
  
template<> inline DataType type<double>() {
  return DataType(DataType::Float64);
}

template<> inline DataType type<std::complex<float>>() {
  return DataType(DataType::Complex64);
}

template<> inline DataType type<std::complex<double>>() {
  return DataType(DataType::Complex128);
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
  /// Create a tensor shape.
  Shape(std::initializer_list<Dimension> dimensions);

  /// Create a tensor shape.
  Shape(std::vector<Dimension> dimensions);

  /// Returns the number of dimensions in the shape.
  size_t numDimensions() const;

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
  /// Create a tensor type.
  Type(DataType, Shape);

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
