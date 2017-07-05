#include "taco/storage/array.h"

#include <cstring>
#include <iostream>

#include "taco/type.h"
#include "taco/error.h"
#include "taco/util/uncopyable.h"
#include "taco/util/strings.h"

using namespace std;

namespace taco {
namespace storage {

struct Array::Content : util::Uncopyable {
  Type   type;
  void*  data;
  size_t size;
  Policy policy = Array::UserOwns;

  ~Content() {
    switch (policy) {
      case UserOwns:
        // do nothing
        break;
      case Free:
        free(data);
        break;
      case Delete:
        switch (type.getKind()) {
          case Type::Bool:
            delete[] ((bool*)data);
            break;
          case Type::UInt:
            switch (type.getNumBits()) {
              case 8:
                delete[] ((uint8_t*)data);
                break;
              case 16:
                delete[] ((uint16_t*)data);
                break;
              case 32:
                delete[] ((uint32_t*)data);
                break;
              case 64:
                delete[] ((uint64_t*)data);
                break;
            }
            break;
          case Type::Int:
            switch (type.getNumBits()) {
            switch (type.getNumBits()) {
              case 8:
                delete[] ((int8_t*)data);
                break;
              case 16:
                delete[] ((int16_t*)data);
                break;
              case 32:
                delete[] ((int32_t*)data);
                break;
              case 64:
                delete[] ((int64_t*)data);
                break;
            }
            }
            break;
          case Type::Float:
            switch (type.getNumBits()) {
              case 32:
                delete[] ((float*)data);
                break;
              case 64:
                delete[] ((double*)data);
                break;
            }
            break;
          case Type::Undefined:
            taco_ierror;
            break;
        }
        break;
    }
  }
};

Array::Array() : content(new Content) {
}

Array::Array(Type type, void* data, size_t size, Policy policy) : Array() {
  content->type = type;
  content->data = data;
  content->size = size;
  content->policy = policy;
}

const Type& Array::getType() const {
  return content->type;
}

size_t Array::getSize() const {
  return content->size;
}

const void* Array::getData() const {
  return content->data;
}

void* Array::getData() {
  return content->data;
}

void Array::zero() {
  memset(getData(), 0, getSize() * getType().getNumBytes());
}

template<typename T>
void printData(ostream& os, const Array& array) {
  const T* data = static_cast<const T*>(array.getData());
  os << "[";
  if (array.getSize() > 0) {
    os << data[0];
  }
  for (size_t i = 1; i < array.getSize(); i++) {
    os << ", " << data[i];
  }
  os << "]";
}

std::ostream& operator<<(std::ostream& os, const Array& array) {
  Type type = array.getType();
  switch (type.getKind()) {
    case Type::Bool:
      printData<bool>(os, array);
      break;
    case Type::UInt:
      switch (type.getNumBits()) {
        case 8:
          printData<uint8_t>(os, array);
          break;
        case 16:
          printData<uint16_t>(os, array);
          break;
        case 32:
          printData<uint32_t>(os, array);
          break;
        case 64:
          printData<uint64_t>(os, array);
          break;
      }
      break;
    case Type::Int:
      switch (type.getNumBits()) {
        case 8:
          printData<int8_t>(os, array);
          break;
        case 16:
          printData<int16_t>(os, array);
          break;
        case 32:
          printData<int32_t>(os, array);
          break;
        case 64:
          printData<int64_t>(os, array);
          break;
      }
      break;
    case Type::Float:
      switch (type.getNumBits()) {
        case 32:
          printData<float>(os, array);
          break;
        case 64:
          printData<double>(os, array);
          break;
      }
      break;
    case Type::Undefined:
      os << "[]";
      break;
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, Array::Policy policy) {
  switch (policy) {
    case Array::UserOwns:
      os << "user";
      break;
    case Array::Free:
      os << "free";
      break;
    case Array::Delete:
      os << "delete";
      break;
  }
  return os;
}

}}
