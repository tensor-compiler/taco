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
  string head, matrix, formats, field, symmetry;
  lineStream >> head >> matrix >> formats >> field >> symmetry;
  taco_uassert(head=="%%MatrixMarket") << "Unknown header of MatrixMarket";
  taco_uassert(matrix=="matrix")       << "Unknown header of MatrixMarket";
  // formats = [coordinate array]
  taco_uassert(formats=="coordinate")  << "MatrixMarket format not available";
  // field = [real integer complex pattern]
  taco_uassert(field=="real")          << "MatrixMarket field not available";
  // symmetry = [general symmetric skew-symmetric Hermitian]
  taco_uassert(symmetry=="general")    << "MatrixMarket symmetry not available";
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

  // The first non-comment line is the header with dimension sizes and nnz
  char* linePtr = (char*)line.data();
  size_t rows = strtoul(linePtr, &linePtr, 10);
  size_t cols = strtoul(linePtr, &linePtr, 10);
  size_t nnz = strtoul(linePtr, &linePtr, 10);
  taco_uassert(rows <= INT_MAX) << "Number of rows in file exceeds INT_MAX";
  taco_uassert(cols <= INT_MAX) << "Number of columns in file exceeds INT_MAX";
  taco_uassert(nnz <= INT_MAX) << "Number of non-zeros in file exceeds INT_MAX";

  vector<int> coordinates;
  vector<double> values;
  coordinates.reserve(nnz*2);
  values.reserve(nnz);

  while (std::getline(stream, line)) {
    linePtr = (char*)line.data();
    long rowIdx = strtol(linePtr, &linePtr, 10);
    long colIdx = strtol(linePtr, &linePtr, 10);
    double val = strtod(linePtr, &linePtr);
    taco_uassert(rowIdx <= INT_MAX && colIdx <= INT_MAX) <<
        "Coordinate in file is larger than INT_MAX";

    coordinates.push_back(rowIdx);
    coordinates.push_back(colIdx);
    values.push_back(val);
  }

  // Create matrix
  TensorBase tensor(ComponentType::Double, {(int)rows,(int)cols}, format);
  tensor.reserve(nnz);

  // Insert coordinates
  for (size_t i = 0; i < nnz; i++) {
    tensor.insert({coordinates[i*2]-1, coordinates[i*2+1]-1}, values[i]);
  }

  if (pack) {
    tensor.pack();
  }

  return tensor;
}

void write(std::string filename, const TensorBase& tensor) {
  taco_iassert(tensor.getOrder() == 2) <<
      "The .mtx format only supports matrices. Consider using the .tns format "
      "instead";

  std::ofstream file;
  file.open(filename);
  taco_uassert(file.is_open()) << "Error opening file: " << filename;
  write(file, tensor);
  file.close();
}

void write(std::ostream& stream, const TensorBase& tensor) {
  taco_iassert(tensor.getOrder() == 2) <<
      "The .mtx format only supports matrices. Consider using the .tns format "
      "instead";

  stream << "%% MatrixMarket matrix coordinate real general" << std::endl;
  stream << "%"                                              << std::endl;
  stream << util::join(tensor.getDimensions(), " ") << " ";
  stream << tensor.getStorage().getSize().numValues() << endl;
  for (auto& value : iterate<double>(tensor)) {
    for (int coord : value.first) {
      stream << coord+1 << " ";
    }
    stream << value.second << endl;
  }
}


}}}
