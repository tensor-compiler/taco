#include "taco/io/tns_file_format.h"

#include <iostream>
#include <sstream>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <string>

#include "taco/tensor_base.h"
#include "taco/util/error.h"

using namespace std;

namespace taco {
namespace io {
namespace tns {

static void insertFirstComponent(std::ifstream &tnsfile, TensorBase *tensor,
                          std::vector<int> &dims) {
  std::string line;
  if (!std::getline(tnsfile, line)) {
    return;
  }

  std::stringstream iss(line);
  std::string tok;
  std::vector<std::string> tokens;

  while (iss >> tok) {
    tokens.push_back(tok);
  }

  if (tensor->getOrder() + 1 != tokens.size()) {
    dims.clear();
    return;
  }

  std::vector<int> coord(tensor->getOrder());
  for (size_t i = 0; i < tensor->getOrder(); ++i) {
    coord[i] = std::stoi(tokens[i]) - 1;
    dims[i] = std::max(dims[i], coord[i] + 1);
  }

  double val = std::stod(tokens.back());
  if (val != 0.0) {
    tensor->insert(coord, val);
  }
}

void readFile(std::ifstream &tnsfile, std::vector<int> &dims,
              TensorBase *tensor) {
  dims = tensor->getDimensions(); 

  // Insert first non-zero component into tensor, verifying that TNS file 
  // describes tensor of expected order.
  insertFirstComponent(tnsfile, tensor, dims);

  // If TNS file describes tensor of different order, then quit loading.
  if (dims.size() != tensor->getOrder()) {
    return;
  }

  std::string line;
  std::string tok;

  std::vector<int> coord(tensor->getOrder());

  // Insert remaining non-zero components into tensor.
  while (std::getline(tnsfile, line)) {
    std::stringstream iss(line);
    
    for (size_t i = 0; i < tensor->getOrder(); ++i) {
      iss >> tok;
      coord[i] = std::stoi(tok) - 1;
      dims[i] = std::max(dims[i], coord[i] + 1);
    }
    
    iss >> tok;

    double val = std::stod(tok);
    if (val != 0.0) {
      tensor->insert(coord, val);
    }
  }

  tensor->pack();
}

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

TensorBase readTensor(std::string filename, std::string name) {
  std::ifstream file;
  file.open(filename);
  taco_uassert(file.is_open()) << "Error opening file: " << filename;
  TensorBase tensor = readTensor(file, name);
  file.close();
  return tensor;
}

TensorBase readTensor(std::istream& stream, std::string name) {
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
      coordinate[i] = idx;
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
      coordinate[j] = coordinates[i*3 + j];
    }
    tensor.insert(coordinate, values[i]);
  }

  return tensor;
}

}}}
