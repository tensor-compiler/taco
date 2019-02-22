#include "taco/storage/file_io_tns.h"

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <cmath>
#include <climits>

#include "taco/tensor.h"
#include "taco/format.h"
#include "taco/error.h"
#include "taco/storage/pack.h"
#include "taco/util/strings.h"
#include "taco/util/files.h"

using namespace std;

namespace taco {


// TensorBase read functions ---

template <typename T>
TensorBase dispatchReadTNS(std::string filename, const T& format, bool pack) {
  std::fstream file;
  util::openStream(file, filename, fstream::in);
  TensorBase tensor = readTNS(file, format, pack);
  file.close();
  return tensor;
}

TensorBase readTNS(std::string filename, const ModeFormat& modetype, bool pack) {
  return dispatchReadTNS(filename, modetype, pack);
}

TensorBase readTNS(std::string filename, const Format& format, bool pack) {
  return dispatchReadTNS(filename, format, pack);
}

template <typename T>
TensorBase dispatchReadTNS(std::istream& stream, const T& format, bool pack) {
  std::vector<int>    coordinates;
  std::vector<double> values;

  std::string line;
  if (!std::getline(stream, line)) {
    return TensorBase();
  }

  // Infer tensor order from the first coordinate
  vector<string> toks = util::split(line, " ");
  size_t order = toks.size()-1;
  std::vector<int> dimensions(order);
  std::vector<int> coordinate(order);

  // Load data
  do {
    char* linePtr = (char*)line.data();
    for (size_t i = 0; i < order; i++) {
      long idx = strtol(linePtr, &linePtr, 10);
      taco_uassert(idx <= INT_MAX)<<"Coordinate in file is larger than INT_MAX";
      coordinate[i] = (int)idx - 1;
      dimensions[i] = std::max(dimensions[i], (int)idx);
    }
    coordinates.insert(coordinates.end(), coordinate.begin(), coordinate.end());
    double val = strtod(linePtr, &linePtr);
    values.push_back(val);

  } while (std::getline(stream, line));

  // Create tensor
  const size_t nnz = values.size();
  TensorBase tensor(type<double>(), dimensions, format);
  tensor.reserve(nnz);

  // Insert coordinates (TODO add and use bulk insertion)
  for (size_t i = 0; i < nnz; i++) {
    for (size_t j = 0; j < order; j++) {
      coordinate[j] = coordinates[i*order + j];
    }
    tensor.insert(coordinate, values[i]);
  }

  cout << values[3] << endl;

  if (pack) {
    tensor.pack();
  }

  return tensor;
}

TensorBase readTNS(std::istream& stream, const ModeFormat& modetype, bool pack) {
  return dispatchReadTNS(stream, modetype, pack);
}

TensorBase readTNS(std::istream& stream, const Format& format, bool pack) {
  return dispatchReadTNS(stream, format, pack);
}

// TensorStorage read functions --

template <typename T>
TensorStorage dispatchReadTNS(std::string filename, const T& format) {
  std::fstream file;
  util::openStream(file, filename, fstream::in);
  TensorStorage storage = readToStorageTNS(file, format);
  file.close();
  return storage;
}

TensorStorage readToStorageTNS(std::string filename, const ModeFormat& modetype) {
  return dispatchReadTNS(filename, modetype);
}

TensorStorage readToStorageTNS(std::string filename, const Format& format) {
  return dispatchReadTNS(filename, format);
}

template <typename T>
TensorStorage dispatchReadTNS(std::istream& stream, const T& format) {
  std::vector<int>    coordinates;
  std::vector<double> values;

  std::string line;
  bool streamIsEmpty = !std::getline(stream, line);
  taco_uassert(!streamIsEmpty) << "The provided input stream is empty. Can't generate a TensorStorage object.";

  // Infer tensor order from the first coordinate
  vector<string> toks = util::split(line, " ");
  size_t order = toks.size() - 1;
  std::vector<int> dimensions(order);
  std::vector<int> coordinate(order);

  // Load data
  do {
    char* linePtr = (char*)line.data();
    for (size_t i = 0; i < order; i++) {
      long idx = strtol(linePtr, &linePtr, 10);
      taco_uassert(idx <= INT_MAX) << "Coordinate in file is larger than INT_MAX";
      coordinate[i] = (int)idx - 1;
      dimensions[i] = std::max(dimensions[i], (int)idx);
    }
    coordinates.insert(coordinates.end(), coordinate.begin(), coordinate.end());
    double val = strtod(linePtr, &linePtr);
    values.push_back(val);

  } while (std::getline(stream, line));

  // Create tensorStorage
  TensorStorage storage(type<double>(), dimensions, format);
  const size_t nnz = values.size();
  std::vector<TypedIndexVector> coords(order, TypedIndexVector(type<double>(), nnz));

  for (size_t i = 0; i < nnz; i++) {
    for (size_t j = 0; j < order; j++) {
      coords[j][i] = coordinates[i * order + j];
    }
  }
  cout << values[3] << endl;
  return pack(type<double>(), dimensions, format, coords, values.data());
}

TensorStorage readToStorageTNS(std::istream& stream, const ModeFormat& modetype) {
  return dispatchReadTNS(stream, modetype);
}

TensorStorage readToStorageTNS(std::istream& stream, const Format& format) {
  return dispatchReadTNS(stream, format);
}

// Write functions --

void writeTNS(std::string filename, const TensorBase& tensor) {
  std::fstream file;
  util::openStream(file, filename, fstream::out);
  writeTNS(file, tensor);
  file.close();
}

void writeTNS(std::ostream& stream, const TensorBase& tensor) {
  for (auto& value : iterate<double>(tensor)) {
    for (size_t coord : value.first) {
      stream << coord+1 << " ";
    }
    stream << value.second << endl;
  }
}

void writeFromStorageTNS(std::string filename, const TensorStorage& storage) {
  std::fstream file;
  util::openStream(file, filename, fstream::out);
  writeFromStorageTNS(file, storage);
  file.close();
}

void writeFromStorageTNS(std::ostream& stream, const TensorStorage& storage) {
  for (auto& value : storage.iterator<size_t,double>()) {
    for (size_t coord : value.first) {
      stream << coord+1 << " ";
    }
    stream << value.second << endl;
  }
}

}
