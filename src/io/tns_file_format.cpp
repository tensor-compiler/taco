#include "taco/io/tns_file_format.h"

#include <iostream>
#include <sstream>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <string>
#include <cmath>

#include "taco/tensor_base.h"
#include "taco/util/error.h"

using namespace std;

namespace taco {
namespace io {
namespace tns {

void writeFile(std::ofstream &tnsfile, std::string name, 
               const TensorBase *tensor) {
  for (const auto& val : *tensor) {
    for (size_t i = 0; i < val.loc.size(); ++i) {
      tnsfile << val.loc[i] + 1 << " " ;
    }
    tnsfile << val.dval << ((std::floor(val.dval) == val.dval) ? ".0 " : " ") 
            << std::endl;
  }
}

TensorBase read(std::string filename, std::string name) {
  std::ifstream file;
  file.open(filename);
  taco_uassert(file.is_open()) << "Error opening file: " << filename;
  TensorBase tensor = read(file, name);
  file.close();
  return tensor;
}

TensorBase read(std::istream& stream, std::string name) {
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
  TensorBase tensor(name, ComponentType::Double, dimensions);
  tensor.reserve(nnz);

  // Insert coordinates (TODO add and use bulk insertion)
  for (size_t i = 0; i < nnz; i++) {
    for (size_t j = 0; j < order; j++) {
      coordinate[j] = coordinates[i*order + j];
    }
    tensor.insert(coordinate, values[i]);
  }

  return tensor;
}

void write(std::string filename, const TensorBase& tensor) {
}

void write(std::ostream& stream, const TensorBase& tensor) {
}

}}}
