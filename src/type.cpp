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
    case Type::UInt:
      switch (bits) {
        case 1: case 8: case 16: case 32: case 64:
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
        case 32: case 64:
          return true;
      }
      break;
  }
  return false;
}

Type::Type(Kind kind) : kind(kind) {
  switch (kind) {
    case UInt:
      bits = sizeof(unsigned int)*8;
      break;
    case Int:
      bits = sizeof(int)*8;
      break;
    case Float:
      bits = sizeof(double)*8;
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
  return os << type.getKind() << type.getNumBits();
}

std::ostream& operator<<(std::ostream& os, const Type::Kind& kind) {
  switch (kind) {
    case Type::UInt:
      os << "uint";
      break;
    case Type::Int:
      os << "int";
      break;
    case Type::Float:
      os << "float";
      break;
  }
  return os;
}

bool operator==(const Type&, const Type&);
bool operator!=(const Type&, const Type&);

}
