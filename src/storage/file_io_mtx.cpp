#include "taco/storage/file_io_mtx.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <climits>

#include "taco/tensor.h"
#include "taco/format.h"
#include "taco/error.h"
#include "taco/storage/pack.h"
#include "taco/util/strings.h"
#include "taco/util/timers.h"
#include "taco/util/files.h"

using namespace std;

namespace taco {

// TensorBase read functions ---

template <typename T>
TensorBase dispatchReadMTX(std::string filename, const T& format, bool pack) {
  std::fstream file;
  util::openStream(file, filename, fstream::in);
  TensorBase tensor = readMTX(file, format, pack);
  file.close();
  return tensor;
}

TensorBase readMTX(std::string filename, const ModeFormat& modetype, bool pack) {
  return dispatchReadMTX(filename, modetype, pack);
}

TensorBase readMTX(std::string filename, const Format& format, bool pack) {
  return dispatchReadMTX(filename, format, pack);
}

template <typename T>
TensorBase dispatchReadMTX(std::istream& stream, const T& format, bool pack) {
  string line;
  if (!std::getline(stream, line)) {
    return TensorBase();
  }

  // Read Header
  std::stringstream lineStream(line);
  string head, type, formats, field, symmetry;
  lineStream >> head >> type >> formats >> field >> symmetry;
  taco_uassert(head=="%%MatrixMarket") << "Unknown header of MatrixMarket";
  // type = [matrix tensor]
  taco_uassert((type=="matrix") || (type=="tensor"))
                                       << "Unknown type of MatrixMarket";
  // formats = [coordinate array]
  // field = [real integer complex pattern]
  taco_uassert(field=="real")          << "MatrixMarket field not available";
  // symmetry = [general symmetric skew-symmetric Hermitian]
  taco_uassert((symmetry=="general") || (symmetry=="symmetric"))
                                       << "MatrixMarket symmetry not available";

  bool symm = (symmetry=="symmetric");

  TensorBase tensor;
  if (formats=="coordinate")
    tensor = readSparse(stream,format,symm);
  else if (formats=="array")
    tensor = readDense(stream,format,symm);
  else
    taco_uerror << "MatrixMarket format not available";

  if (pack) {
    tensor.pack();
  }

  return tensor;
}

TensorBase readMTX(std::istream& stream, const ModeFormat& modetype, bool pack) {
  return dispatchReadMTX(stream, modetype, pack);
}

TensorBase readMTX(std::istream& stream, const Format& format, bool pack) {
  return dispatchReadMTX(stream, format, pack);
}

template <typename T>
TensorBase dispatchReadSparse(std::istream& stream, const T& format, 
                              bool symm) {
  string line;
  std::getline(stream,line);

  // Skip comments at the top of the file
  string token;
  do {
    std::stringstream lineStream(line);
    lineStream >> token;
    if (token[0] != '%') {
      break;
    }
  } while (std::getline(stream, line));

  // The first non-comment line is the header with dimensions
  vector<int> dimensions;
  char* linePtr = (char*)line.data();
  while (size_t dimension = strtoul(linePtr, &linePtr, 10)) {
    taco_uassert(dimension <= INT_MAX) << "Dimension exceeds INT_MAX";
    dimensions.push_back(static_cast<int>(dimension));
  }
  size_t nnz = dimensions[dimensions.size()-1];
  dimensions.pop_back();
  if (symm)
    taco_uassert(dimensions.size()==2) << "Symmetry only available for matrix";

  vector<int> coordinates;
  vector<double> values;
  coordinates.reserve(nnz*dimensions.size());
  values.reserve(nnz);

  while (std::getline(stream, line)) {
    linePtr = (char*)line.data();
    for (size_t i=0; i < dimensions.size(); i++) {
      long index = strtol(linePtr, &linePtr, 10);
      taco_uassert(index <= INT_MAX) << "Index exceeds INT_MAX";
      coordinates.push_back(static_cast<int>(index));
    }
    double val = strtod(linePtr, &linePtr);
    values.push_back(val);
  }

  // Create matrix
  TensorBase tensor(type<double>(), dimensions, format);
  if (symm)
    tensor.reserve(2*nnz);
  else
    tensor.reserve(nnz);

  // Insert coordinates
  std::vector<int> coord;
  for (size_t i = 0; i < nnz; i++) {
    coord.clear();
    for (size_t mode = 0; mode < dimensions.size(); mode++) {
      coord.push_back(coordinates[i*dimensions.size() + mode] -1);
    }
    tensor.insert(coord, values[i]);
    if (symm && coord.front() != coord.back()) {
      std::reverse(coord.begin(), coord.end());
      tensor.insert(coord, values[i]);
    }
  }

  return tensor;
}

TensorBase readSparse(std::istream& stream, const ModeFormat& modetype, 
                      bool symm) {
  return dispatchReadSparse(stream, modetype, symm);
}

TensorBase readSparse(std::istream& stream, const Format& format, bool symm) {
  return dispatchReadSparse(stream, format, symm);
}

template <typename T>
TensorBase dispatchReadDense(std::istream& stream, const T& format, bool symm) {
  string line;
  std::getline(stream,line);

  // Skip comments at the top of the file
  string token;
  do {
    std::stringstream lineStream(line);
    lineStream >> token;
    if (token[0] != '%') {
      break;
    }
  } while (std::getline(stream, line));

  // The first non-comment line is the header with dimension sizes
  vector<int> dimensions;
  char* linePtr = (char*)line.data();
  while (size_t dimension = strtoul(linePtr, &linePtr, 10)) {
    taco_uassert(dimension <= INT_MAX) << "Dimension exceeds INT_MAX";
    dimensions.push_back(static_cast<int>(dimension));
  }
  if (symm)
    taco_uassert(dimensions.size()==2) << "Symmetry only available for matrix";

  vector<double> values;
  auto size = std::accumulate(begin(dimensions), end(dimensions),
                              1, std::multiplies<double>());
  values.reserve(size);

  while (std::getline(stream, line)) {
    linePtr = (char*)line.data();
    double val = strtod(linePtr, &linePtr);
    values.push_back(val);
  }

  // Create matrix
  TensorBase tensor(type<double>(), dimensions, format);
  if (symm)
    tensor.reserve(2*size);
  else
    tensor.reserve(size);

  // Insert coordinates
  std::vector<int> coord;
  for (auto n = 0; n<size; n++) {
    coord.clear();
    auto index=n;
    for (size_t mode = 0; mode < dimensions.size()-1; mode++) {
      coord.push_back(index%dimensions[mode]);
      index=index/dimensions[mode];
    }
    coord.push_back(index);
    tensor.insert(coord, values[n]);
    if (symm && coord.front() != coord.back()) {
      std::reverse(coord.begin(), coord.end());
      tensor.insert(coord, values[n]);
    }
  }

  return tensor;
}

TensorBase readDense(std::istream& stream, const ModeFormat& modetype, 
                     bool symm) {
  return dispatchReadDense(stream, modetype, symm);
}

TensorBase readDense(std::istream& stream, const Format& format, bool symm) {
  return dispatchReadDense(stream, format, symm);
}

// TensorStorage read functions ---

template <typename T>
TensorStorage dispatchReadMTX(std::string filename, const T& format) {
  std::fstream file;
  util::openStream(file, filename, fstream::in);
  TensorStorage storage = readToStorageMTX(file, format);
  file.close();
  return storage;
}

TensorStorage readToStorageMTX(std::string filename, const ModeFormat& modetype) {
  return dispatchReadMTX(filename, modetype);
}

TensorStorage readToStorageMTX(std::string filename, const Format& format) {
  return dispatchReadMTX(filename, format);
}

template <typename T>
TensorStorage dispatchReadMTX(std::istream& stream, const T& format) {
  string line;
  bool streamIsEmpty = !std::getline(stream, line);
  taco_uassert(streamIsEmpty) << "The provided input stream is empty. Can't generate a TensorStorage object.";

  // Read Header
  std::stringstream lineStream(line);
  string head, type, formats, field, symmetry;
  lineStream >> head >> type >> formats >> field >> symmetry;
  taco_uassert(head=="%%MatrixMarket") << "Unknown header of MatrixMarket";
  // type = [matrix tensor]
  taco_uassert((type=="matrix") || (type=="tensor"))
                                       << "Unknown type of MatrixMarket";
  // formats = [coordinate array]
  // field = [real integer complex pattern]
  taco_uassert(field=="real")          << "MatrixMarket field not available";
  // symmetry = [general symmetric skew-symmetric Hermitian]
  taco_uassert((symmetry=="general") || (symmetry=="symmetric"))
                                       << "MatrixMarket symmetry not available";

  bool symm = (symmetry=="symmetric");

  if (formats=="coordinate")
    return readToStorageSparse(stream,format,symm);
  else if (formats=="array")
    return readToStorageDense(stream,format,symm);
  else
    taco_uerror << "MatrixMarket format not available";
    return TensorStorage(Datatype::Undefined, std::vector<int>(), Format());
}

TensorStorage readToStorageMTX(std::istream& stream, const ModeFormat& modetype) {
  return dispatchReadMTX(stream, modetype);
}

TensorStorage readToStorageMTX(std::istream& stream, const Format& format) {
  return dispatchReadMTX(stream, format);
}

template <typename T>
TensorStorage dispatchReadToStorageSparse(std::istream& stream, const T& format, 
                              bool symm) {
  string line;
  std::getline(stream,line);

  // Skip comments at the top of the file
  string token;
  do {
    std::stringstream lineStream(line);
    lineStream >> token;
    if (token[0] != '%') {
      break;
    }
  } while (std::getline(stream, line));

  // The first non-comment line is the header with dimensions
  vector<int> dimensions;
  char* linePtr = (char*)line.data();
  while (size_t dimension = strtoul(linePtr, &linePtr, 10)) {
    taco_uassert(dimension <= INT_MAX) << "Dimension exceeds INT_MAX";
    dimensions.push_back(static_cast<int>(dimension));
  }
  size_t nnz = dimensions[dimensions.size()-1];
  dimensions.pop_back();
  if (symm)
    taco_uassert(dimensions.size()==2) << "Symmetry only available for matrix";

  vector<int> coordinates;
  vector<double> values;
  coordinates.reserve(nnz*dimensions.size());
  values.reserve(nnz);

  while (std::getline(stream, line)) {
    linePtr = (char*)line.data();
    for (size_t i=0; i < dimensions.size(); i++) {
      long index = strtol(linePtr, &linePtr, 10);
      taco_uassert(index <= INT_MAX) << "Index exceeds INT_MAX";
      coordinates.push_back(static_cast<int>(index));
    }
    double val = strtod(linePtr, &linePtr);
    values.push_back(val);
  }

  size_t order = dimensions.size();

  // Create matrix
  TensorStorage storage(type<double>(), dimensions, format);
  std::vector<TypedIndexVector> coords(order, TypedIndexVector(type<double>(), nnz * (1 + symm)));

  // Insert coordinates
  std::vector<int> coord;
  size_t symmOffset = 0;
  for (size_t i = 0; i < nnz; i++) {
    coord.clear();
    for (size_t mode = 0; mode < order; mode++) {
      coord.push_back(coordinates[i * order + mode] -1);
      coords[mode][i] = coord[mode];
    }
    // Symm value
    if (symm && coord.front() != coord.back()) {
      std::reverse(coord.begin(), coord.end());
      for (size_t mode = 0; mode < order; mode++) {
        coords[mode][nnz + symmOffset] = coord[mode];
      }
      symmOffset++;
      values.push_back(values[i]);
    }
  }
  for (size_t mode = 0; mode < order; mode++) {
    coords[mode].resize(nnz + symmOffset);
  }

  return pack(type<double>(), dimensions, format, coords, values.data());
}

TensorStorage readToStorageSparse(std::istream& stream, const ModeFormat& modetype, 
                      bool symm) {
  return dispatchReadToStorageSparse(stream, modetype, symm);
}

TensorStorage readToStorageSparse(std::istream& stream, const Format& format, bool symm) {
  return dispatchReadToStorageSparse(stream, format, symm);
}

template <typename T>
TensorStorage dispatchReadToStorageDense(std::istream& stream, const T& format, bool symm) {
  string line;
  std::getline(stream,line);

  // Skip comments at the top of the file
  string token;
  do {
    std::stringstream lineStream(line);
    lineStream >> token;
    if (token[0] != '%') {
      break;
    }
  } while (std::getline(stream, line));

  // The first non-comment line is the header with dimension sizes
  vector<int> dimensions;
  char* linePtr = (char*)line.data();
  while (size_t dimension = strtoul(linePtr, &linePtr, 10)) {
    taco_uassert(dimension <= INT_MAX) << "Dimension exceeds INT_MAX";
    dimensions.push_back(static_cast<int>(dimension));
  }
  if (symm)
    taco_uassert(dimensions.size()==2) << "Symmetry only available for matrix";

  vector<double> values;
  auto size = std::accumulate(begin(dimensions), end(dimensions),
                              1, std::multiplies<double>());
  values.reserve(size);

  while (std::getline(stream, line)) {
    linePtr = (char*)line.data();
    double val = strtod(linePtr, &linePtr);
    values.push_back(val);
  }

  size_t order = dimensions.size();

  TensorStorage storage(type<double>(), dimensions, format);
  std::vector<TypedIndexVector> coords(order, TypedIndexVector(type<double>(), size * (1 + symm)));

  // Insert coordinates
  std::vector<int> coord;
  size_t symmOffset = 0;
  for (auto i = 0; i < size; i++) {
    coord.clear();
    auto index = i;
    for (size_t mode = 0; mode < order - 1; mode++) {
      coord.push_back(index % dimensions[mode]);
      index = index / dimensions[mode];
    }
    coord.push_back(index);
    for (size_t mode = 0; mode < order; mode++) {
      coords[mode][i] = coord[mode];
    }

    // Insert Symm value
    if (symm && coord.front() != coord.back()) {
      std::reverse(coord.begin(), coord.end());
      for (size_t mode = 0; mode < order; mode++) {
        coords[mode][size + symmOffset] = coord[mode];
      }
      symmOffset++;
      values.push_back(values[i]);
    }
  }
  for (size_t mode = 0; mode < order; mode++) {
    coords[mode].resize(size + symmOffset);
  }

  return pack(type<double>(), dimensions, format, coords, values.data());
}

TensorStorage readToStorageDense(std::istream& stream, const ModeFormat& modetype, 
                        bool symm) {
  return dispatchReadToStorageDense(stream, modetype, symm);
}

TensorStorage readToStorageDense(std::istream& stream, const Format& format, bool symm) {
  return dispatchReadToStorageDense(stream, format, symm);
}

// TensorBase write functions   ---

void writeMTX(std::string filename, const TensorBase& tensor) {
  std::fstream file;
  util::openStream(file, filename, fstream::out);
  writeMTX(file, tensor);
  file.close();
}

void writeMTX(std::ostream& stream, const TensorBase& tensor) {
  if (isDense(tensor.getFormat()))
    writeDense(stream, tensor);
  else
    writeSparse(stream, tensor);
}

void writeSparse(std::ostream& stream, const TensorBase& tensor) {
  if(tensor.getOrder() == 2)
    stream << "%%MatrixMarket matrix coordinate real general" << std::endl;
  else
    stream << "%%MatrixMarket tensor coordinate real general" << std::endl;
  stream << "%"                                             << std::endl;
  stream << util::join(tensor.getDimensions(), " ") << " ";
  stream << tensor.getStorage().getIndex().getSize() << endl;
  for (auto& value : iterate<double>(tensor)) {
    for (size_t coord : value.first) {
      stream << coord+1 << " ";
    }
    stream << value.second << endl;
  }
}

void writeDense(std::ostream& stream, const TensorBase& tensor) {
  if(tensor.getOrder() == 2)
    stream << "%%MatrixMarket matrix array real general" << std::endl;
  else
    stream << "%%MatrixMarket tensor array real general" << std::endl;
  stream << "%"                                        << std::endl;
  stream << util::join(tensor.getDimensions(), " ") << " " << endl;
  for (auto& value : iterate<double>(tensor)) {
    stream << value.second << endl;
  }
}

void writeFromStorageMTX(std::string filename, const TensorStorage& storage) {
  std::fstream file;
  util::openStream(file, filename, fstream::out);
  writeFromStorageMTX(file, storage);
  file.close();
}

void writeFromStorageMTX(std::ostream& stream, const TensorStorage& storage) {
  if (isDense(storage.getFormat()))
    writeFromStorageDense(stream, storage);
  else
    writeFromStorageSparse(stream, storage);
}

void writeFromStorageSparse(std::ostream& stream, const TensorStorage& storage) {
  if(storage.getOrder() == 2)
    stream << "%%MatrixMarket matrix coordinate real general" << std::endl;
  else
    stream << "%%MatrixMarket tensor coordinate real general" << std::endl;
  stream << "%"                                             << std::endl;
  stream << util::join(storage.getDimensions(), " ") << " ";
  stream << storage.getIndex().getSize() << endl;
  for (auto& value : storage.iterator<size_t,double>()) {
    for (size_t coord : value.first) {
      stream << coord+1 << " ";
    }
    stream << value.second << endl;
  }
}

void writeFromStorageDense(std::ostream& stream, const TensorStorage& storage) {
  if(storage.getOrder() == 2)
    stream << "%%MatrixMarket matrix array real general" << std::endl;
  else
    stream << "%%MatrixMarket tensor array real general" << std::endl;
  stream << "%"                                        << std::endl;
  stream << util::join(storage.getDimensions(), " ") << " " << endl;
  for (auto& value : storage.iterator<size_t,double>()) {
    stream << value.second << endl;
  }
}

}
