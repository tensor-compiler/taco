#include "taco/storage/storage.h"

#include <iostream>
#include <string>

#include "taco/type.h"
#include "taco/format.h"
#include "taco/error.h"
#include "taco/storage/index.h"
#include "taco/storage/array.h"
#include "taco/storage/array_util.h"
#include "taco/util/strings.h"

using namespace std;

namespace taco {
namespace storage {

// class Storage
struct Storage::Content {
  DataType      componentType;
  vector<int>   dimensions;
  Format        format;

  taco_tensor_t tensorData;

  Index         index;
  Array         values;

  Content(DataType componentType, vector<int> dimensions, Format format)
      : componentType(componentType), dimensions(dimensions), format(format) {
        size_t order = dimensions.size();

        taco_iassert(order <= INT_MAX && componentType.getNumBits() <= INT_MAX);
        vector<int32_t> dimensionsInt32(order);
        vector<int32_t> modeOrdering(order);
        vector<taco_mode_t> modeTypes(order);
        for (size_t i=0; i < order; ++i) {
          dimensionsInt32[i] = dimensions[i];
          modeOrdering[i] = format.getModeOrdering()[i];
          auto modeType  = format.getModeTypes()[i];
          if (modeType == Dense) {
            modeTypes[i] = taco_mode_dense;
          } else if (modeType == Sparse) {
            modeTypes[i] = taco_mode_sparse;
          } else {
            taco_not_supported_yet;
          }
        }

        init_taco_tensor_t(&tensorData, order, componentType.getNumBits(),
                           dimensionsInt32.data(), modeOrdering.data(),
                           modeTypes.data());
  }

  ~Content() {
    deinit_taco_tensor_t(&tensorData);
  }
};

Storage::Storage() : content(nullptr) {
}

Storage::Storage(DataType componentType, const vector<int>& dimensions,
                 Format format)
    : content(new Content(componentType, dimensions, format)) {
}

const Format& Storage::getFormat() const {
  return content->format;
}

DataType Storage::getComponentType() const {
  return content->componentType;
}

const vector<int>& Storage::getDimensions() const {
  return content->dimensions;
}

const Index& Storage::getIndex() const {
  return content->index;
}

Index Storage::getIndex() {
  return content->index;
}

const Array& Storage::getValues() const {
  return content->values;
}

Array Storage::getValues() {
  return content->values;
}

size_t Storage::getSizeInBytes() {
  size_t indexSizeInBytes = 0;
  const auto& index = getIndex();
  for (size_t i = 0; i < index.numModeIndices(); i++) {
    const auto& modeIndex = index.getModeIndex(i);
    for (size_t j = 0; j < modeIndex.numIndexArrays(); j++) {
      const auto& indexArray = modeIndex.getIndexArray(j);
      indexSizeInBytes += indexArray.getSize() *
                          indexArray.getType().getNumBytes();
    }
  }
  const auto& values = getValues();
  return indexSizeInBytes + values.getSize() * values.getType().getNumBytes();
}

Storage::operator struct taco_tensor_t*() const {
  taco_tensor_t* tensorData = &content->tensorData;

  DataType ctype = getComponentType();
  size_t order = getDimensions().size();
  Format format = getFormat();
  Index index = getIndex();

  for (size_t i = 0; i < order; i++) {
    auto modeType  = format.getModeTypes()[i];
    auto modeIndex = index.getModeIndex(i);

    if (modeType == Dense) {
      const Array& size = modeIndex.getIndexArray(0);
      tensorData->indices[i][0] = (uint8_t*)size.getData();
    } else if (modeType == Sparse) {
      // Results for assemblies don't have sparse indices
      if (modeIndex.numIndexArrays() > 0) {
        const Array& pos = modeIndex.getIndexArray(0);
        const Array& idx = modeIndex.getIndexArray(1);
        tensorData->indices[i][0] = (uint8_t*)pos.getData();
        tensorData->indices[i][1] = (uint8_t*)idx.getData();
      }
    } else {
      taco_not_supported_yet;
    }
  }

  taco_iassert(ctype.getNumBits() <= INT_MAX);
  tensorData->vals  = (uint8_t*)getValues().getData();

  return &content->tensorData;
}

void Storage::setIndex(const Index& index) {
  content->index = index;
}

void Storage::setValues(const Array& values) {
  content->values = values;
}

std::ostream& operator<<(std::ostream& os, const Storage& storage) {
  return os << storage.getIndex() << endl << storage.getValues();
}

}}
