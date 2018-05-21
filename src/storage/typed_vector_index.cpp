#include "taco/storage/typed_vector_index.h"
namespace taco {
namespace storage {
  //class TypedIndexVector
TypedIndexVector::TypedIndexVector() : type(DataType::Undefined) {
}

TypedIndexVector::TypedIndexVector(DataType type) : type(type){
}

TypedIndexVector::TypedIndexVector(DataType type, size_t size) : type(type){
  resize(size);
}

void TypedIndexVector::push_back(void *value) {
  resize(size() + 1);
  set(size() - 1, value);
}

void TypedIndexVector::push_back(TypedIndexVal value) {
  taco_iassert(value.getType() == type);
  resize(size() + 1);
  set(size() - 1, value);
}

void TypedIndexVector::push_back(TypedIndexRef value) {
  taco_iassert(value.getType() == type);
  resize(size() + 1);
  set(size() - 1, value);
}

void TypedIndexVector::push_back_vector(TypedIndexVector vector) {
  resize(size() + vector.size());
  memcpy(&get(size()-vector.size()).get(), vector.data(), type.getNumBytes()*vector.size());
}

void TypedIndexVector::resize(size_t size) {
  charVector.resize(size * type.getNumBytes());
}

TypedIndexRef TypedIndexVector::get(size_t index) const {
  return TypedIndexRef(getType(), (void *) &charVector[index * type.getNumBytes()]);
}

void TypedIndexVector::set(size_t index, TypedIndexVal value) {
  taco_iassert(value.getType() == type);
  get(index) = value;
}

void TypedIndexVector::set(size_t index, TypedIndexRef value) {
  taco_iassert(value.getType() == type);
  get(index) = value;
}

void TypedIndexVector::clear() {
  charVector.clear();
}

size_t TypedIndexVector::size() const {
  return charVector.size() / type.getNumBytes();
}

char* TypedIndexVector::data() const {
  return (char *) charVector.data();
}

DataType TypedIndexVector::getType() const {
  return type;
}

bool TypedIndexVector::operator==(const TypedIndexVector &other) const {
  if (size() != other.size()) return false;
  return (memcmp(data(), other.data(), size()*type.getNumBytes()) == 0);
}

bool TypedIndexVector::operator!=(const TypedIndexVector &other) const {
  return !(*this == other);
}

//lexicographical comparison
bool TypedIndexVector::operator<(const TypedIndexVector &other) const {
  size_t minSize = size() < other.size() ? size() : other.size();
  for (size_t i = 0; i < minSize; i++) {
    if (get(i) < other.get(i)) return true;
    if (get(i) > other.get(i)) return false;
  }
  return size() < other.size();
}

bool TypedIndexVector::operator>(const TypedIndexVector &other) const {
  return !(*this < other) && !(*this == other);
}


TypedIndexRef TypedIndexVector::operator[] (const size_t index) const {
  return get(index);
}

TypedIndexVector::iterator TypedIndexVector::begin() {
  return iterator(&get(0), type);
}

TypedIndexVector::iterator TypedIndexVector::end() {
  return iterator(&get(size()), type);
}

TypedIndexVector::const_iterator TypedIndexVector::begin() const
{
  return const_iterator(&get(0), type);
}

TypedIndexVector::const_iterator TypedIndexVector::end() const
{
  return const_iterator(&get(size()), type);
}

}
}
