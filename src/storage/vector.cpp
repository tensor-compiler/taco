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

void TypedVector::push_back(TypedRef value) {
  taco_iassert(value.getType() == type);
  resize(size() + 1);
  set(size() - 1, value);
}

void TypedVector::push_back_vector(TypedVector vector) {
  resize(size() + vector.size());
  memcpy(&get(size()-vector.size()).get(), vector.data(), type.getNumBytes()*vector.size());
}

void TypedVector::resize(size_t size) {
  charVector.resize(size * type.getNumBytes());
}

TypedRef TypedVector::get(size_t index) const {
  return TypedRef(getType(), (void *) &charVector[index * type.getNumBytes()]);
}

void TypedVector::set(size_t index, TypedValue value) {
  taco_iassert(value.getType() == type);
  get(index) = value;
}

void TypedVector::set(size_t index, TypedRef value) {
  taco_iassert(value.getType() == type);
  get(index) = value;
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

//lexicographical comparison
bool TypedVector::operator<(const TypedVector &other) const {
  size_t minSize = size() < other.size() ? size() : other.size();
  for (size_t i = 0; i < minSize; i++) {
    if (get(i) < other.get(i)) return true;
    if (get(i) > other.get(i)) return false;
  }
  return size() < other.size();
}

bool TypedVector::operator>(const TypedVector &other) const {
  return !(*this < other) && !(*this == other);
}


TypedRef TypedVector::operator[] (const size_t index) const {
  return get(index);
}

TypedVector::iterator TypedVector::begin() {
  return iterator(&get(0), type);
}

TypedVector::iterator TypedVector::end() {
  return iterator(&get(size()), type);
}

TypedVector::const_iterator TypedVector::begin() const
{
  return const_iterator(&get(0), type);
}

TypedVector::const_iterator TypedVector::end() const
{
  return const_iterator(&get(size()), type);
}

}
}
