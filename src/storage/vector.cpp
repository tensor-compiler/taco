#include "taco/storage/vector.h"
namespace taco {
namespace storage {
  //class TypedVector
TypedVector::TypedVector() : type(DataType::Undefined) {
}

TypedVector::TypedVector(DataType type) : type(type){
}

TypedVector::TypedVector(DataType type, size_t size) : type(type){
  resize(size);
}

void TypedVector::push_back(void *value) {
  resize(size() + 1);
  set(size() - 1, value);
}

void TypedVector::push_back_vector(TypedVector vector) {
  resize(size() + vector.size());
  memcpy(get(size()-vector.size()), vector.data(), type.getNumBytes()*vector.size());
}

void TypedVector::resize(size_t size) {
  charVector.resize(size * type.getNumBytes());
}

void* TypedVector::get(int index) const {
  return (void *) &charVector[index * type.getNumBytes()];
}

void TypedVector::get(int index, void *result) const {
  memcpy(result, get(index), type.getNumBytes());
}

void TypedVector::set(int index, void *result) {
  memcpy(get(index), result, type.getNumBytes());
}

void TypedVector::clear() {
  charVector.clear();
}

size_t TypedVector::size() const {
  return charVector.size() / type.getNumBytes();
}

char* TypedVector::data() const {
  return (char *) charVector.data();
}

DataType TypedVector::getType() const {
  return type;
}

bool TypedVector::operator==(TypedVector &other) const {
  if (size() != other.size()) return false;
  return (memcmp(data(), other.data(), size()*type.getNumBytes()) == 0);
}

bool TypedVector::operator!=(TypedVector &other) const {
  return !(*this == other);
}
}
}
