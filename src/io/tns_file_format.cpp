#include "taco/io/tns_file_format.h"

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <cmath>
#include <limits.h>

#include "taco/tensor.h"
#include "taco/format.h"
#include "taco/error.h"
#include "taco/util/strings.h"

using namespace std;

namespace taco {
namespace io {
namespace tns {

TensorBase read(std::string filename, const Format& format, bool pack) {
  std::ifstream file;
  file.open(filename);
  taco_uassert(file.is_open()) << "Error opening file: " << filename;
  TensorBase tensor = read(file, format, pack);
  file.close();
  return tensor;
}

TensorBase read(std::istream& stream, const Format& format, bool pack) {
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

  if (pack) {
    tensor.pack();
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
  for (auto& value : iterate<double>(tensor)) {
    for (int coord : value.first) {
      stream << coord+1 << " ";
    }
    stream << value.second << endl;
  }
}

}}}
