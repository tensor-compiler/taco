#ifndef SRC_IO_MTX_FILE_FORMAT_H_
#define SRC_IO_MTX_FILE_FORMAT_H_

#include <fstream>
#include <vector>

namespace taco {
class TensorBase;

namespace io {
namespace mtx {

void readFile(std::ifstream &mtxfile,
              int* nrow, int* ncol, int* nnzero,
              TensorBase* tensor);

void writeFile(std::ofstream &mtxfile, std::string name,
               const std::vector<int> dimensions, int nnzero);

}}}

#endif /* SRC_IO_MTX_FILE_FORMAT_H_ */
