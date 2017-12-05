#include "taco/type.h"

#include "error/error_messages.h"
#include "taco/util/collections.h"

#include <ostream>
#include <set>
#include <complex>

using namespace std;

namespace taco {

static bool supportedBitWidth(DataType::Kind kind, size_t bits) {
  switch (kind) {
    case DataType::Bool:
      if (bits == sizeof(bool)) return true;
      break;
    case DataType::UInt:
      switch (bits) {
        case 8: case 16: case 32: case 64:
          return true;
      }
      break;
    case DataType::Int:
      switch (bits) {
        case 8: case 16: case 32: case 64:
          return true;
      }
      break;
    case DataType::Float:
      switch (bits) {
        case 32:
          taco_iassert(sizeof(float) == 4) << "fp assumption broken";
          return true;
        case 64:
          taco_iassert(sizeof(double) == 8) << "fp assumption broken";
          return true;
      }
      break;
    case DataType::Undefined:
      taco_ierror;
      break;
  }
  return false;
}

DataType::DataType() : kind(Undefined) {
}

DataType::DataType(Kind kind) : kind(kind) {
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

DataType::DataType(Kind kind, size_t bits) : kind(kind), bits(bits) {
  taco_uassert(supportedBitWidth(kind, bits)) <<
      error::type_bitwidt << " (" << kind << ": " << bits << ")";
}

DataType::Kind DataType::getKind() const {
  return this->kind;
}

bool DataType::isBool() const {
  return getKind() == Bool;
}

bool DataType::isUInt() const {
  return getKind() == UInt;
}

bool DataType::isInt() const {
  return getKind() == Int;
}

bool DataType::isFloat() const {
  return getKind() == Float;
}

size_t DataType::getNumBytes() const {
  return (getNumBits() + 7) / 8;
}

size_t DataType::getNumBits() const {
  return this->bits;
}

std::ostream& operator<<(std::ostream& os, const DataType& type) {
  switch (type.getKind()) {
    case DataType::Bool:
      os << "bool";
      break;
    case DataType::UInt:
      os << "uint" << type.getNumBits() << "_t";
      break;
    case DataType::Int:
      os << "int" << type.getNumBits() << "_t";
      break;
    case DataType::Float:
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
    case DataType::Undefined:
      os << "Undefined";
      break;
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const DataType::Kind& kind) {
  switch (kind) {
    case DataType::Bool:
      os << "Bool";
      break;
    case DataType::UInt:
      os << "UInt";
      break;
    case DataType::Int:
      os << "Int";
      break;
    case DataType::Float:
      os << "Float";
      break;
    case DataType::Undefined:
      os << "Undefined";
      break;
  }
  return os;
}

bool operator==(const DataType& a, const DataType& b) {
  return a.getKind() == b.getKind() && a.getNumBits() == b.getNumBits();
}

bool operator!=(const DataType& a, const DataType& b) {
  return a.getKind() != b.getKind() || a.getNumBits() != b.getNumBits();
}

DataType Bool(size_t bits) {
  return DataType(DataType::Bool, bits);
}

DataType Int(size_t bits) {
  return DataType(DataType::Int, bits);
}

DataType UInt(size_t bits) {
  return DataType(DataType::UInt, bits);
}

DataType Float(size_t bits) {
  return DataType(DataType::Float, bits);
}

}
