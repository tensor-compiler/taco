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

void TypedVector::push_back(TypedValue value) {
  taco_iassert(value.getType() == type);
  resize(size() + 1);
  set(size() - 1, value);
}

void TypedVector::push_back_vector(TypedVector vector) {
  resize(size() + vector.size());
  memcpy(get(size()-vector.size()).get(), vector.data(), type.getNumBytes()*vector.size());
}

void TypedVector::resize(size_t size) {
  charVector.resize(size * type.getNumBytes());
}

TypedValue TypedVector::get(int index) const {
  return TypedValue(getType(), (void *) &charVector[index * type.getNumBytes()]);
}

void TypedVector::copyTo(int index, void *location) const {
  memcpy(location, get(index).get(), type.getNumBytes());
}

void TypedVector::set(int index, void *value) {
  memcpy(get(index).get(), value, type.getNumBytes());
}

void TypedVector::set(int index, TypedValue value) {
  taco_iassert(value.getType() == type);
  memcpy(get(index).get(), value.get(), type.getNumBytes());
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

bool TypedVector::operator==(const TypedVector &other) const {
  if (size() != other.size()) return false;
  return (memcmp(data(), other.data(), size()*type.getNumBytes()) == 0);
}

bool TypedVector::operator!=(const TypedVector &other) const {
  return !(*this == other);
}

TypedValue TypedVector::operator[] (const int index) const {
  return get(index);
}

TypedVector::iterator TypedVector::begin() {
  return iterator((TypedValue *) get(0).get(), type);
}

TypedVector::iterator TypedVector::end() {
  return iterator((TypedValue *) get(size()).get(), type);
}

TypedVector::const_iterator TypedVector::begin() const
{
  return const_iterator((TypedValue *) get(0).get(), type);
}

TypedVector::const_iterator TypedVector::end() const
{
  return const_iterator((TypedValue *) get(size()).get(), type);
}

}
}
