#include "taco/storage/storage.h"

#include <iostream>
#include <string>
#include <climits>

#include "taco/type.h"
#include "taco/format.h"
#include "taco/error.h"
#include "taco/storage/index.h"
#include "taco/storage/array.h"
#include "taco/util/strings.h"

using namespace std;

namespace taco {

// class Storage
struct TensorStorage::Content {
  Datatype      componentType;
  vector<int>   dimensions;
  Format        format;

  taco_tensor_t *tensorData;

  Index         index;
  Array         values;

  Content(Datatype componentType, vector<int> dimensions, Format format)
      : componentType(componentType), dimensions(dimensions), format(format),
        index(format) {
    int order = (int)dimensions.size();

    taco_iassert(order <= INT_MAX && componentType.getNumBits() <= INT_MAX);
    vector<int32_t> dimensionsInt32(order);
    vector<int32_t> modeOrdering(order);
    vector<taco_mode_t> modeTypes(order);
    for (int i=0; i < order; ++i) {
      dimensionsInt32[i] = dimensions[i];
      modeOrdering[i] = format.getModeOrdering()[i];
      auto modeType  = format.getModeFormats()[i];
      if (modeType.getName() == Dense.getName()) {
        modeTypes[i] = taco_mode_dense;
      } else if (modeType.getName() == Sparse.getName()) {
        modeTypes[i] = taco_mode_sparse;
      } else if (modeType.getName() == Singleton.getName()) {
        modeTypes[i] = taco_mode_sparse;
      } else {
        taco_not_supported_yet;
      }
    }

    tensorData = init_taco_tensor_t(order, componentType.getNumBits(),
                       dimensionsInt32.data(), modeOrdering.data(),
                       modeTypes.data());
  }

  ~Content() {
    deinit_taco_tensor_t(tensorData);
  }
};

TensorStorage::TensorStorage(Datatype componentType,
                             const vector<int>& dimensions, Format format)
    : content(new Content(componentType, dimensions, format)) {
}

const Format& TensorStorage::getFormat() const {
  return content->format;
}

Datatype TensorStorage::getComponentType() const {
  return content->componentType;
}

int TensorStorage::getOrder() const {
  return getFormat().getOrder();
}

const vector<int>& TensorStorage::getDimensions() const {
  return content->dimensions;
}

const Index& TensorStorage::getIndex() const {
  return content->index;
}

Index TensorStorage::getIndex() {
  return content->index;
}

const Array& TensorStorage::getValues() const {
  return content->values;
}

Array TensorStorage::getValues() {
  return content->values;
}

size_t TensorStorage::getSizeInBytes() {
  size_t indexSizeInBytes = 0;
  const auto& index = getIndex();
  for (int i = 0; i < index.numModeIndices(); i++) {
    const auto& modeIndex = index.getModeIndex(i);
    for (int j = 0; j < modeIndex.numIndexArrays(); j++) {
      const auto& indexArray = modeIndex.getIndexArray(j);
      indexSizeInBytes += indexArray.getSize() *
                          indexArray.getType().getNumBytes();
    }
  }
  const auto& values = getValues();
  return indexSizeInBytes + values.getSize() * values.getType().getNumBytes();
}

TensorStorage::operator struct taco_tensor_t*() const {
  taco_tensor_t* tensorData = content->tensorData;

  taco_iassert(getComponentType().getNumBits() <= INT_MAX);
  int order = getOrder();
  Format format = getFormat();
  Index index = getIndex();

  for (int i = 0; i < order; i++) {
    auto modeType  = format.getModeFormats()[i];
    auto modeIndex = index.getModeIndex(i);

    // Dense modes don't have indices (they iterate over mode sizes)
    if (modeType.getName() == Dense.getName()) {
      // TODO Uncomment assertion and remove code in this conditional
      // taco_iassert(modeIndex.numIndexArrays() == 0)
      //     << modeIndex.numIndexArrays();
      const Array& size = modeIndex.getIndexArray(0);
      tensorData->indices[i][0] = (uint8_t*)size.getData();
    }
    // Sparse levels have two indices (pos and idx)
    else if (modeType.getName() == Sparse.getName()) {
      // TODO Uncomment assert and remove conditional
      // taco_iassert(modeIndex.numIndexArrays() == 2)
      //     << modeIndex.numIndexArrays();
      if (modeIndex.numIndexArrays() > 0) {
        const Array& pos = modeIndex.getIndexArray(0);
        const Array& idx = modeIndex.getIndexArray(1);
        tensorData->indices[i][0] = (uint8_t*)pos.getData();
        tensorData->indices[i][1] = (uint8_t*)idx.getData();
      }
    }
    else if (modeType.getName() == Singleton.getName()) {
      // TODO Uncomment assert and remove conditional
      // taco_iassert(modeIndex.numIndexArrays() == 2)
      //     << modeIndex.numIndexArrays();
      if (modeIndex.numIndexArrays() > 0) {
        const Array& idx = modeIndex.getIndexArray(1);
        tensorData->indices[i][1] = (uint8_t*)idx.getData();
      }
    }
    else {
      taco_not_supported_yet;
    }
  }

  tensorData->vals  = (uint8_t*)getValues().getData();

  return content->tensorData;
}

void TensorStorage::setIndex(const Index& index) {
  content->index = index;
}

void TensorStorage::setValues(const Array& values) {
  content->values = values;
}

bool equals(TensorStorage a, TensorStorage b) {
  return false;
}

std::ostream& operator<<(std::ostream& os, const TensorStorage& storage) {
  if (storage.getOrder() > 0) {
    os << storage.getIndex() << std::endl;
  }
  return os << storage.getValues();
}

}
