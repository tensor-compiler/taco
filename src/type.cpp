#include "taco/type.h"

#include "error/error_messages.h"
#include "taco/util/collections.h"

#include <ostream>
#include <set>
#include <complex>

using namespace std;

namespace taco {

static bool supportedBitWidth(Type::Kind kind, size_t bits) {
  switch (kind) {
    case Type::Bool:
      if (bits == sizeof(bool)) return true;
      break;
    case Type::UInt:
      switch (bits) {
        case 8: case 16: case 32: case 64:
          return true;
      }
      break;
    case Type::Int:
      switch (bits) {
        case 8: case 16: case 32: case 64:
          return true;
      }
      break;
    case Type::Float:
      switch (bits) {
        case 32:
          taco_iassert(sizeof(float) == 4) << "fp assumption broken";
          return true;
        case 64:
          taco_iassert(sizeof(double) == 8) << "fp assumption broken";
          return true;
      }
      break;
    case Type::Undefined:
      taco_ierror;
      break;
  }
  return false;
}

Type::Type() : kind(Undefined) {
}

Type::Type(Kind kind) : kind(kind) {
  switch (kind) {
    case Bool:
      bits = sizeof(bool);
      break;
    case UInt:
      bits = sizeof(unsigned int)*8;
      break;
    case Int:
      bits = sizeof(int)*8;
      break;
    case Float:
      bits = sizeof(double)*8;
      break;
    case Undefined:
      taco_uerror << "use default constructor to construct an undefined type";
      break;
  }
}

Type::Type(Kind kind, size_t bits) : kind(kind), bits(bits) {
  taco_uassert(supportedBitWidth(kind, bits)) <<
      error::type_bitwidt << " (" << kind << ": " << bits << ")";
}

Type::Kind Type::getKind() const {
  return this->kind;
}

bool Type::isBool() const {
  return getKind() == Bool;
}

bool Type::isUInt() const {
  return getKind() == UInt;
}

bool Type::isInt() const {
  return getKind() == Int;
}

bool Type::isFloat() const {
  return getKind() == Float;
}

size_t Type::getNumBytes() const {
  return (getNumBits() + 7) / 8;
}

size_t Type::getNumBits() const {
  return this->bits;
}

std::ostream& operator<<(std::ostream& os, const Type& type) {
  switch (type.getKind()) {
    case Type::Bool:
      os << "bool";
      break;
    case Type::UInt:
      os << "uint" << type.getNumBits() << "_t";
      break;
    case Type::Int:
      os << "int" << type.getNumBits() << "_t";
      break;
    case Type::Float:
      switch (type.getNumBits()) {
        case 32:
          taco_iassert(sizeof(float) == 4);
          os << "float";
          break;
        case 64:
          taco_iassert(sizeof(double) == 8);
          os << "double";
          break;
        default:
          taco_ierror << "unsupported float bit width: " << type.getNumBits();
          break;
      }
      break;
    case Type::Undefined:
      os << "Undefined";
      break;
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const Type::Kind& kind) {
  switch (kind) {
    case Type::Bool:
      os << "Bool";
      break;
    case Type::UInt:
      os << "UInt";
      break;
    case Type::Int:
      os << "Int";
      break;
    case Type::Float:
      os << "Float";
      break;
    case Type::Undefined:
      os << "Undefined";
      break;
  }
  return os;
}

bool operator==(const Type& a, const Type& b) {
  return a.getKind() == b.getKind() && a.getNumBits() == b.getNumBits();
}

bool operator!=(const Type& a, const Type& b) {
  return a.getKind() != b.getKind() || a.getNumBits() != b.getNumBits();
}

Type Bool(size_t bits) {
  return Type(Type::Bool, bits);
}

Type Int(size_t bits) {
  return Type(Type::Int, bits);
}

Type UInt(size_t bits) {
  return Type(Type::UInt, bits);
}

Type Float(size_t bits) {
  return Type(Type::Float, bits);
}

}
