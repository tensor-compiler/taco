#include "mtx_file_format.h"

#include <iostream>
#include <sstream>
#include <cstdlib>

#include "error.h"

namespace taco {
namespace io {
namespace mtx {

void readFile(std::ifstream &mtxfile,
              int* nrow, int* ncol, int* nnzero,
              internal::Tensor* tensor) {

  std::string line;
  int rowind,colind;
  double value;
  std::string val;
  while(std::getline(mtxfile,line)) {
    std::stringstream iss(line);
    char firstChar;
    iss >> firstChar;
    // Skip comments
    if (firstChar != '%') {
      iss.clear();
      iss.str(line);
      iss >> *nrow >> *ncol >> *nnzero;
      break;
    }
  }

  while(std::getline(mtxfile,line)) {
    std::stringstream iss(line);
    iss >> rowind >> colind >> val;
    value = std::stod(val);
    tensor->insert({rowind-1,colind-1},value);
  }
  tensor->pack();
}

}}}
