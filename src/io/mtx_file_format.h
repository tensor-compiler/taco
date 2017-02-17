#ifndef SRC_IO_MTX_FILE_FORMAT_H_
#define SRC_IO_MTX_FILE_FORMAT_H_

#include <fstream>

#include "internal_tensor.h"

namespace taco {
namespace io {
namespace mtx {

void readFile(std::ifstream &mtxfile,
              int* nrow, int* ncol, int* nnzero,
              internal::Tensor* tensor);

void writeFile(std::ofstream &mtxfile, std::string name,
               int nrow, int ncol, int nnzero);

}}}

#endif /* SRC_IO_MTX_FILE_FORMAT_H_ */
