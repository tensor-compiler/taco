#include "taco/storage/storage.h"

#include <iostream>
#include <string>
#include <climits>

#include "taco/type.h"
#include "taco/format.h"
#include "taco/error.h"
#include "taco/storage/array.h"
#include "taco/storage/file_io_tns.h"
#include "taco/storage/file_io_mtx.h"
#include "taco/storage/file_io_rb.h"
#include "taco/storage/index.h"
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
      if (modeType == Dense) {
        modeTypes[i] = taco_mode_dense;
      } else if (modeType == Sparse) {
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

TensorStorage::TensorStorage(Datatype componentType, const std::vector<int>& dimensions,
                ModeFormat modeType)
    : TensorStorage(componentType, dimensions,
                    std::vector<ModeFormatPack>(dimensions.size(), modeType)) {
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
    if (modeType == Dense) {
      // TODO Uncomment assertion and remove code in this conditional
      // taco_iassert(modeIndex.numIndexArrays() == 0)
      //     << modeIndex.numIndexArrays();
      const Array& size = modeIndex.getIndexArray(0);
      tensorData->indices[i][0] = (uint8_t*)size.getData();
    }
    // Sparse levels have two indices (pos and idx)
    else if (modeType == Sparse) {
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

// File IO

static string getExtension(string filename) {
  return filename.substr(filename.find_last_of(".") + 1);
}

template <typename T, typename U>
TensorStorage dispatchRead(T& file, FileType filetype, U format) {
  switch (filetype) {
    case FileType::ttx:
    case FileType::mtx:
      return readToStorageMTX(file, format);
      break;
    case FileType::tns:
      return readToStorageTNS(file, format);
      break;
    case FileType::rb:
      return readToStorageRB(file, format);
      break;
  }
}

template <typename U>
TensorStorage dispatchRead(std::string filename, U format) {
  string extension = getExtension(filename);

  if (extension == "ttx") {
    return dispatchRead(filename, FileType::ttx, format);
  }
  else if (extension == "tns") {
    return dispatchRead(filename, FileType::tns, format);
  }
  else if (extension == "mtx") {
    return dispatchRead(filename, FileType::mtx, format);
  }
  else if (extension == "rb") {
    return dispatchRead(filename, FileType::rb, format);
  }
  else {
    taco_uerror << "File extension not recognized: " << filename << std::endl;
    return TensorStorage(Datatype::Undefined, std::vector<int>(), Format());
  }
}

TensorStorage readToStorage(std::string filename, ModeFormat modetype) {
  return dispatchRead(filename, modetype);
}

TensorStorage readToStorage(std::string filename, Format format) {
  return dispatchRead(filename, format);
}

TensorStorage readToStorage(string filename, FileType filetype, ModeFormat modetype) {
  return dispatchRead(filename, filetype, modetype);
}

TensorStorage readToStorage(string filename, FileType filetype, Format format) {
  return dispatchRead(filename, filetype, format);
}

TensorStorage readToStorage(istream& stream, FileType filetype, ModeFormat modetype) {
  return dispatchRead(stream, filetype, modetype);
}

TensorStorage readToStorage(istream& stream, FileType filetype, Format format) {
  return dispatchRead(stream, filetype, format);
}

void dispatchWrite(string file, const TensorStorage& storage, FileType filetype) {
  switch (filetype) {
    case FileType::ttx:
    case FileType::mtx:
      writeFromStorageMTX(file, storage);
      break;
    case FileType::tns:
      writeFromStorageTNS(file, storage);
      break;
    case FileType::rb:
      writeFromStorageRB(file, storage);
      break;
  }
}

void dispatchWrite(ostream& file, const TensorStorage& storage, FileType filetype) {
  switch (filetype) {
    case FileType::ttx:
    case FileType::mtx:
      writeFromStorageMTX(file, storage);
      break;
    case FileType::tns:
      writeFromStorageTNS(file, storage);
      break;
    case FileType::rb:
      writeFromStorageRB(file, storage);
      break;
  }
}

void writeFromStorage(string filename, const TensorStorage& storage) {
  string extension = getExtension(filename);
  if (extension == "ttx") {
    dispatchWrite(filename, storage, FileType::ttx);
  }
  else if (extension == "tns") {
    dispatchWrite(filename, storage, FileType::tns);
  }
  else if (extension == "mtx") {
    taco_iassert(storage.getOrder() == 2) <<
       "The .mtx format only supports matrices. Consider using the .ttx format "
       "instead";
    dispatchWrite(filename, storage, FileType::mtx);
  }
  else if (extension == "rb") {
    dispatchWrite(filename, storage, FileType::rb);
  }
  else {
    taco_uerror << "File extension not recognized: " << filename << std::endl;
  }
}

void writeFromStorage(string filename, FileType filetype, const TensorStorage& storage) {
  dispatchWrite(filename, storage, filetype);
}

void writeFromStorage(ostream& stream, FileType filetype, const TensorStorage& storage) {
  dispatchWrite(stream, storage, filetype);
}

}
