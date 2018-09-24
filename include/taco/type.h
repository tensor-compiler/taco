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
class Datatype {
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
    Undefined  /// Undefined type
  };

  /// Construct an undefined type.
  Datatype();

  /// Construct a taco basic type with default bit widths.
  Datatype(Kind);

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
  int getNumBytes() const;

  /// Returns the number of bits required to store one element of this type.
  int getNumBits() const;

private:
  Kind kind;
};

std::ostream& operator<<(std::ostream&, const Datatype&);
std::ostream& operator<<(std::ostream&, const Datatype::Kind&);
bool operator==(const Datatype& a, const Datatype& b);
bool operator!=(const Datatype& a, const Datatype& b);

extern Datatype Bool;
Datatype UInt(int bits = sizeof(unsigned int)*8);
extern Datatype UInt8;
extern Datatype UInt16;
extern Datatype UInt32;
extern Datatype UInt64;
extern Datatype UInt128;
Datatype Int(int bits = sizeof(int)*8);
extern Datatype Int8;
extern Datatype Int16;
extern Datatype Int32;
extern Datatype Int64;
extern Datatype Int128;
Datatype Float(int bits = sizeof(double)*8);
extern Datatype Float32;
extern Datatype Float64;
Datatype Complex(int bits);
extern Datatype Complex64;
extern Datatype Complex128;

Datatype max_type(Datatype a, Datatype b);

template<typename T> inline Datatype type() {
  taco_ierror << "Unsupported type";
  return Int32;
}
  
template<> inline Datatype type<bool>() {
  return Bool;
}

template<> inline Datatype type<unsigned char>() {
  return UInt(sizeof(char)*8);
}
  
template<> inline Datatype type<unsigned short>() {
  return UInt(sizeof(short)*8);
}
  
template<> inline Datatype type<unsigned int>() {
  return UInt(sizeof(int)*8);
}
  
template<> inline Datatype type<unsigned long>() {
  return UInt(sizeof(long)*8);
}
  
template<> inline Datatype type<unsigned long long>() {
  return UInt(sizeof(long long)*8);
}

template<> inline Datatype type<char>() {
  return Int(sizeof(char)*8);
}
  
template<> inline Datatype type<short>() {
  return Int(sizeof(short)*8);
}
  
template<> inline Datatype type<int>() {
  return Int(sizeof(int)*8);
}

template<> inline Datatype type<long>() {
  return Int(sizeof(long)*8);
}
  
template<> inline Datatype type<long long>() {
  return Int(sizeof(long long)*8);
}
  
template<> inline Datatype type<int8_t>() {
  return Int8;
}

template<> inline Datatype type<float>() {
  return Float32;
}
  
template<> inline Datatype type<double>() {
  return Float64;
}

template<> inline Datatype type<std::complex<float>>() {
  return Complex64;
}

template<> inline Datatype type<std::complex<double>>() {
  return Complex128;
}

/// A union storing all of the different types that a component can take.
union ComponentTypeUnion {
  bool boolValue;

  uint8_t uint8Value;
  uint16_t uint16Value;
  uint32_t uint32Value;
  uint64_t uint64Value;
  unsigned long long uint128Value;

  int8_t int8Value;
  int16_t int16Value;
  int32_t int32Value;
  int64_t int64Value;
  long long int128Value;

  float float32Value;
  double float64Value;

  std::complex<float> complex64Value;
  std::complex<double> complex128Value;
  ComponentTypeUnion() {int32Value = 0;}
};

/// A union storing all of the different types that an index can take.
/// 64 bits (instead of 128-bits of ComponentTypeUnion) to avoid performance
/// penalty of allowing for long long types
union IndexTypeUnion {
  uint8_t uint8Value;
  uint16_t uint16Value;
  uint32_t uint32Value;
  uint64_t uint64Value;

  int8_t int8Value;
  int16_t int16Value;
  int32_t int32Value;
  int64_t int64Value;

  IndexTypeUnion() {int32Value = 0;}
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
  int getOrder() const;

  /// Returns the ith dimension.
  Dimension getDimension(size_t i) const;

  /// Iterator to the first dimension.
  std::vector<Dimension>::const_iterator begin() const;

  /// Iterator past the last dimension.
  std::vector<Dimension>::const_iterator end() const;

private:
  std::vector<Dimension> dimensions;
};

bool operator==(const Shape&, const Shape&);
bool operator!=(const Shape&, const Shape&);

/// Print a tensor shape.
std::ostream& operator<<(std::ostream&, const Shape&);


/// A tensor type consists of a shape and a component/data type.
class Type {
public:
  /// Create a default tensor type (double scalar)
  Type();

  /// Create a tensor type.
  Type(Datatype, Shape={});

  Datatype getDataType() const;
  int getOrder() const;
  Shape getShape() const;

private:
  Datatype dtype;
  Shape shape;
};

bool operator==(const Type&, const Type&);
bool operator!=(const Type&, const Type&);

/// Print a tensor type.
std::ostream& operator<<(std::ostream&, const Type&);

/// Check whether the type is a scalar (0-order tensor)
bool isScalar(const Type& type);

}
#endif
