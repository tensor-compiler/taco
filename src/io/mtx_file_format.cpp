#include "taco/io/mtx_file_format.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <climits>

#include "taco/tensor.h"
#include "taco/format.h"
#include "taco/error.h"
#include "taco/util/strings.h"
#include "taco/util/timers.h"

using namespace std;

namespace taco {
namespace io {
namespace mtx {

TensorBase read(std::string filename, const Format& format, bool pack) {
  std::ifstream file;
  file.open(filename);
  taco_uassert(file.is_open()) << "Error opening file: " << filename;
  TensorBase tensor = read(file, format, pack);
  file.close();
  return tensor;
}

TensorBase read(std::istream& stream, const Format& format, bool pack) {
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

  bool symm=false;
  if (symmetry=="symmetric")
    symm = true;

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

TensorBase readSparse(std::istream& stream, const Format& format, bool symm) {
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
  vector<int> dimSizes;
  char* linePtr = (char*)line.data();
  while (int dimSize = strtoul(linePtr, &linePtr, 10)) {
    taco_uassert(dimSize <= INT_MAX) << "Dimension size exceeds INT_MAX";
    dimSizes.push_back(dimSize);
  }
  size_t nnz = dimSizes[dimSizes.size()-1];
  dimSizes.pop_back();
  if (symm)
    taco_uassert(dimSizes.size()==2) << "Symmetry only available for matrix";

  vector<int> coordinates;
  vector<double> values;
  coordinates.reserve(nnz*dimSizes.size());
  values.reserve(nnz);

  while (std::getline(stream, line)) {
    linePtr = (char*)line.data();
    for (size_t i=0; i < dimSizes.size(); i++) {
      long dimIdx = strtol(linePtr, &linePtr, 10);
      coordinates.push_back(dimIdx);
    }
    double val = strtod(linePtr, &linePtr);
    values.push_back(val);
  }

  // Create matrix
  TensorBase tensor(type<double>(), dimSizes, format);
  if (symm)
    tensor.reserve(2*nnz);
  else
    tensor.reserve(nnz);

  // Insert coordinates
  std::vector<int> coord;
  for (size_t i = 0; i < nnz; i++) {
    coord.clear();
    for (size_t dim = 0; dim < dimSizes.size(); dim++) {
      coord.push_back(coordinates[i*dimSizes.size() + dim] -1);
    }
    tensor.insert(coord, values[i]);
    if (symm) {
      std::reverse(coord.begin(), coord.end());
      tensor.insert(coord, values[i]);
    }
  }

  return tensor;
}

TensorBase readDense(std::istream& stream, const Format& format, bool symm) {
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
  vector<int> dimSizes;
  char* linePtr = (char*)line.data();
  while (int dimSize = strtoul(linePtr, &linePtr, 10)) {
    taco_uassert(dimSize <= INT_MAX) << "Dimension size exceeds INT_MAX";
    dimSizes.push_back(dimSize);
  }
  if (symm)
    taco_uassert(dimSizes.size()==2) << "Symmetry only available for matrix";

  vector<double> values;
  auto size = std::accumulate(begin(dimSizes), end(dimSizes),
                              1, std::multiplies<double>());
  values.reserve(size);

  while (std::getline(stream, line)) {
    linePtr = (char*)line.data();
    double val = strtod(linePtr, &linePtr);
    values.push_back(val);
  }

  // Create matrix
  TensorBase tensor(type<double>(), dimSizes, format);
  if (symm)
    tensor.reserve(2*size);
  else
    tensor.reserve(size);

  // Insert coordinates
  std::vector<int> coord;
  for (auto n = 0; n<size; n++) {
    coord.clear();
    auto indice=n;
    for (size_t dim = 0; dim < dimSizes.size()-1; dim++) {
      coord.push_back(indice%dimSizes[dim]);
      indice=indice/dimSizes[dim];
    }
    coord.push_back(indice);
    tensor.insert(coord, values[n]);
    if (symm) {
      std::reverse(coord.begin(), coord.end());
      tensor.insert(coord, values[n]);
    }
  }

  return tensor;
}

void write(std::string filename, const TensorBase& tensor) {
  std::ofstream file;
  file.open(filename);
  taco_uassert(file.is_open()) << "Error opening file: " << filename;
  write(file, tensor);
  file.close();
}

void write(std::ostream& stream, const TensorBase& tensor) {
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
    for (int coord : value.first) {
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
}}}
