#ifndef TACO_STORAGE_STORAGE_H
#define TACO_STORAGE_STORAGE_H

#include <vector>
#include <memory>

#include "taco/format.h"
#include "taco/storage/index.h"
#include "taco/storage/array.h"
#include "taco/storage/typed_vector.h"
#include "taco/storage/typed_index.h"

struct taco_tensor_t;

namespace taco {
class Format;
class Type;
class Datatype;
class Index;
class Array;

/// Storage for a tensor object.  Tensor storage consists of a value array that
/// contains the tensor values and one index per mode.  The type of each
/// mode index is determined by the mode type in the format, and the
/// ordering of the mode indices is determined by the format mode ordering.
class TensorStorage {
public:

  /// Construct tensor storage for the given format.
  TensorStorage(Datatype componentType, const std::vector<int>& dimensions,
                Format format);

  /// Construct tensor storage for the given mode format type.
  TensorStorage(Datatype componentType, const std::vector<int>& dimensions,
                ModeFormat modeType);

  /// Returns the tensor storage format.
  const Format& getFormat() const;

  /// Returns the component type of the tensor storage.
  Datatype getComponentType() const;

  /// Returns the order of the tensor storage.
  int getOrder() const;

  /// Returns the dimensions of the tensor storage.
  const std::vector<int>& getDimensions() const;

  /// Get the tensor index, which describes the non-zero values.
  /// @{
  const Index& getIndex() const;
  Index getIndex();
  /// @}

  /// Returns the value array that contains the tensor components.
  const Array& getValues() const;

  /// Returns the tensor component value array.
  Array getValues();

  /// Returns the size of the storage in bytes.
  size_t getSizeInBytes();

  /// Convert to a taco_tensor_t, whose lifetime is the same as the storage.
  operator struct taco_tensor_t*() const;

  /// Set the tensor index, which describes the non-zero values.
  void setIndex(const Index& index);

  /// Set the tensor component value array.
  void setValues(const Array& values);

  /// Iterator class to iterate over the values stored in the storage object.
  /// Iterates over the values according to the mode order.
  template<typename T, typename CType>
  class const_iterator {
  public:
    typedef const_iterator self_type;
    typedef std::pair<std::vector<T>,CType>  value_type;
    typedef std::pair<std::vector<T>,CType>& reference;
    typedef std::pair<std::vector<T>,CType>* pointer;
    typedef std::forward_iterator_tag iterator_category;

    const_iterator(const const_iterator&) = default;

    const_iterator operator++() {
      advanceIndex();
      return *this;
    }

   const_iterator operator++(int) {
     const_iterator result = *this;
     ++(*this);
     return result;
    }

    const std::pair<std::vector<T>,CType>& operator*() const {
      return curVal;
    }

    const std::pair<std::vector<T>,CType>* operator->() const {
      return &curVal;
    }

    bool operator==(const const_iterator& rhs) {
      return storage == rhs.storage && count == rhs.count;
    }

    bool operator!=(const const_iterator& rhs) {
      return !(*this == rhs);
    }

  private:
    friend class TensorStorage;

    const_iterator(const TensorStorage* storage, bool isEnd = false) :
        storage(storage),
        coord(TypedIndexVector(type<T>(), storage->getOrder())),
        ptrs(TypedIndexVector(type<T>(), storage->getOrder())),
        curVal({std::vector<T>(storage->getOrder()), 0}),
        count(1 + (size_t)isEnd * storage->getIndex().getSize()),
        advance(false) {
      advanceIndex();
    }

    void advanceIndex() {
      advanceIndex(0);
      ++count;
    }

    bool advanceIndex(int lvl) {
      const auto& modeTypes = storage->getFormat().getModeFormats();
      const auto& modeOrdering = storage->getFormat().getModeOrdering();

      if (lvl == storage->getOrder()) {
        if (advance) {
          advance = false;
          return false;
        }

        const TypedIndexVal idx = (lvl == 0) ? TypedIndexVal(type<T>(), 0) : ptrs[lvl - 1];
        curVal.second = ((CType *)storage->getValues().getData())[idx.getAsIndex()];

        for (int i = 0; i < lvl; ++i) {
          const size_t mode = modeOrdering[i];
          curVal.first[mode] = (T)coord[i].getAsIndex();
        }

        advance = true;
        return true;
      }
      
      const auto modeIndex = storage->getIndex().getModeIndex(lvl);

      if (modeTypes[lvl] == Dense) {
        TypedIndexVal size(type<T>(),
                           (int)modeIndex.getIndexArray(0)[0].getAsIndex());
        TypedIndexVal base = ptrs[lvl - 1] * size;
        if (lvl == 0) base.set(0);

        if (advance) {
          goto resume_dense;  // obligatory xkcd: https://xkcd.com/292/
        }

        for (coord[lvl] = 0; coord[lvl] < size; ++coord[lvl]) {
          ptrs[lvl] = base + coord[lvl];

        resume_dense:
          if (advanceIndex(lvl + 1)) {
            return true;
          }
        }
      } else if (modeTypes[lvl] == Sparse) {
        const auto& pos = modeIndex.getIndexArray(0);
        const auto& idx = modeIndex.getIndexArray(1);
        TypedIndexVal k = (lvl == 0) ? TypedIndexVal(type<T>(),0) : ptrs[lvl-1];

        if (advance) {
          goto resume_sparse;
        }

        for (ptrs[lvl] = (int)pos.get((int)k.getAsIndex()).getAsIndex();
             ptrs[lvl] < (int)pos.get((int)k.getAsIndex()+1).getAsIndex();
             ++ptrs[lvl]) {
          coord[lvl] = (int)idx.get((int)ptrs[lvl].getAsIndex()).getAsIndex();

        resume_sparse:
          if (advanceIndex(lvl + 1)) {
            return true;
          }
        }
      } else {
        taco_not_supported_yet;
      }

      return false;
    }

    const TensorStorage*             storage;
    TypedIndexVector                 coord;
    TypedIndexVector                 ptrs;
    std::pair<std::vector<T>,CType>  curVal;
    size_t                           count;
    bool                             advance;
  };

  /// Wrapper to template the index and value types used during
  /// value iteration for performance.
  template<typename T, typename CType>
  class iterator_wrapper {
  public:
    const_iterator<T, CType> begin() const {
      return const_iterator<T, CType>(storage);
    }

    const_iterator<T, CType> end() const {
      return const_iterator<T, CType>(storage, true);
    }

  private:
    friend class TensorStorage;

    iterator_wrapper(const TensorStorage* storage) : storage(storage) { }

    const TensorStorage* storage;
  };

  /// Get an object that can be used to instantiate a foreach loop
  /// to iterate over the values in the storage object.
  /// T:     type of the mode indices
  /// CType: type of the values stored. Must match the component type
  ///        for correct behavior.
  /// Example usage:
  /// for (auto& value : storage.iterator<int, double>()) { ... }
  template<typename T, typename CType>
  iterator_wrapper<T,CType> iterator() const {
    return iterator_wrapper<T,CType>(this);
  }

private:
  struct Content;
  std::shared_ptr<Content> content;
};



/// Compare tensor storage objects.
bool equals(TensorStorage a, TensorStorage b);

/// Print Storage objects to a stream.
std::ostream& operator<<(std::ostream&, const TensorStorage&);

/// The file formats supported by the taco file readers and writers.
enum class FileType {
  /// .tns - The frostt sparse tensor format.  It consists of zero or more
  ///        comment lines preceded by '#', followed by any number of lines with
  ///        one coordinate/value per line.  The tensor dimensions are inferred
  ///        from the largest coordinates.
  tns,

  /// .mtx - The matrix market matrix format.  It consists of a header
  ///        line preceded by '%%', zero or more comment lines preceded by '%',
  ///        a line with the number of rows, the number of columns and the
  //         number of non-zeroes. For sparse matrix and any number of lines
  ///        with one coordinate/value per line, and for dense a list of values.
  mtx,

  /// .ttx - The tensor format derived from matrix market format. It consists
  ///        with the same header file and coordinates/values list.
  ttx,

  /// .rb  - The rutherford-boeing sparse matrix format.
  rb
};

/// Read a tensor from a file. The file format is inferred from the filename
/// and the tensor is returned packed by default.
TensorStorage readToStorage(std::string filename, ModeFormat modeType);

/// Read a tensor from a file. The file format is inferred from the filename
/// and the tensor is returned packed by default.
TensorStorage readToStorage(std::string filename, Format format);

/// Read a tensor from a file of the given file format and the tensor is
/// returned packed by default.
TensorStorage readToStorage(std::string filename, FileType filetype, ModeFormat modetype);

/// Read a tensor from a file of the given file format and the tensor is
/// returned packed by default.
TensorStorage readToStorage(std::string filename, FileType filetype, Format format);

/// Read a tensor from a stream of the given file format. The tensor is returned
/// packed by default.
TensorStorage readToStorage(std::istream& stream, FileType filetype, ModeFormat modetype);

/// Read a tensor from a stream of the given file format. The tensor is returned
/// packed by default.
TensorStorage readToStorage(std::istream& stream, FileType filetype, Format format);

/// Write a tensor to a file. The file format is inferred from the filename.
void writeFromStorage(std::string filename, const TensorStorage& storage);

/// Write a tensor to a file in the given file format.
void writeFromStorage(std::string filename, FileType filetype, const TensorStorage& storage);

/// Write a tensor to a stream in the given file format.
void writeFromStorage(std::ostream& file, FileType filetype, const TensorStorage& storage);

}
#endif
