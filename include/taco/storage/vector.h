#ifndef TACO_STORAGE_VECTOR_H
#define TACO_STORAGE_VECTOR_H
#include <vector>
#include <taco/type.h>
#include <taco/storage/array.h>

namespace taco {
  namespace storage {
    // Like std::vector but for a dynamic DataType type. Backed by a char vector
    class TypedVector {
    public:
      TypedVector();
      TypedVector(DataType type);
      TypedVector(DataType type, size_t size);
      void push_back(void *value);
      void push_back(TypedValue value);
      void push_back_vector(TypedVector vector);
      void resize(size_t size);
      TypedValue get(int index) const;
      void copyTo(int index, void *location) const;
      void set(int index, void *value);
      void set(int index, TypedValue value);
      void clear();
      size_t size() const;
      char* data() const;
      DataType getType() const;
      bool operator==(TypedVector &other) const;
      bool operator!=(TypedVector &other) const;


    private:
      std::vector<char> charVector;
      DataType type;
    };
  }
}
#endif
