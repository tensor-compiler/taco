#include "taco/io/mtx_file_format.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <climits>

#include "taco/tensor_base.h"
#include "taco/util/error.h"
#include "taco/util/strings.h"
#include "taco/util/timers.h"

using namespace std;

namespace taco {
namespace io {
namespace mtx {

TensorBase read(std::string filename, std::string name) {
  std::ifstream file;
  file.open(filename);
  taco_uassert(file.is_open()) << "Error opening file: " << filename;
  TensorBase tensor = read(file, name);
  file.close();
  return tensor;
}

TensorBase read(std::istream& stream, std::string name) {
  string line;
  if (!std::getline(stream, line)) {
    return TensorBase();
  }

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
  TensorBase tensor(name, ComponentType::Double, {(int)rows,(int)cols});
  tensor.reserve(nnz);

  // Insert coordinates
  for (size_t i = 0; i < nnz; i++) {
    tensor.insert({coordinates[i*2]-1, coordinates[i*2+1]-1}, values[i]);
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
  stream << tensor.getStorage().getSize().values << endl;
  for (auto& coord : tensor) {
    for (int loc : coord.loc) {
      stream << loc+1 << " ";
    }
    stream << coord.dval << endl;
  }
}


}}}
