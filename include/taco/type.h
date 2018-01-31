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
    UInt128,
    Int8,
    Int16,
    Int32,
    Int64,
    Int128,
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
DataType UInt(int bits = sizeof(unsigned int)*8);
DataType UInt8();
DataType UInt16();
DataType UInt32();
DataType UInt64();
DataType UInt128();
DataType Int(int bits = sizeof(int)*8);
DataType Int8();
DataType Int16();
DataType Int32();
DataType Int64();
DataType Int128();
DataType Float(int bits = sizeof(double)*8);
DataType Float32();
DataType Float64();
DataType Complex(int bits);
DataType Complex64();
DataType Complex128();
DataType max_type(DataType a, DataType b);

template<typename T> inline DataType type() {
  taco_ierror << "Unsupported type";
  return Int32();
}
  
template<> inline DataType type<bool>() {
  return Bool();
}

template<> inline DataType type<unsigned char>() {
  return UInt(sizeof(char)*8);
}
  
template<> inline DataType type<unsigned short>() {
  return UInt(sizeof(short)*8);
}
  
template<> inline DataType type<unsigned int>() {
  return UInt(sizeof(int)*8);
}
  
template<> inline DataType type<unsigned long>() {
  return UInt(sizeof(long)*8);
}
  
template<> inline DataType type<unsigned long long>() {
  return UInt(sizeof(long long)*8);
}

template<> inline DataType type<char>() {
  return Int(sizeof(char)*8);
}
  
template<> inline DataType type<short>() {
  return Int(sizeof(short)*8);
}
  
template<> inline DataType type<int>() {
  return Int(sizeof(int)*8);
}

template<> inline DataType type<long>() {
  return Int(sizeof(long)*8);
}
  
template<> inline DataType type<long long>() {
  return Int(sizeof(long long)*8);
}
  
template<> inline DataType type<int8_t>() {
  return Int8();
}

template<> inline DataType type<float>() {
  return Float32();
}
  
template<> inline DataType type<double>() {
  return Float64();
}

template<> inline DataType type<std::complex<float>>() {
  return Complex64();
}

template<> inline DataType type<std::complex<double>>() {
  return Complex128();
}

// Like std::vector but for a dynamic DataType type. Backed by a char vector
class TypedVector {
  public:
    TypedVector(DataType type);
    TypedVector(DataType type, size_t size);
    void push_back(void *value);
    void resize(size_t size);
    void get(int index, void *result);
    void set(int index, void *value);
    void clear();
    size_t size();

  private:
    DataType type;
    std::vector<char> charVector;
};

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

bool operator==(const Dimension&, const Dimension&);
bool operator!=(const Dimension&, const Dimension&);

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
