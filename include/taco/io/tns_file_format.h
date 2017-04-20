#ifndef SRC_IO_TNS_FILE_FORMAT_H_
#define SRC_IO_TNS_FILE_FORMAT_H_

#include <fstream>
#include <vector>

namespace taco {
class TensorBase;

namespace io {
namespace tns {

void readFile(std::ifstream &tnsfile, std::vector<int> &dims, 
              TensorBase *tensor);

void writeFile(std::ofstream &mtxfile, std::string name,
               const TensorBase *tensor);

/// Read a tensor from a file.
TensorBase readFile(std::ifstream &tnsfile, std::string name="");

}}}

#endif /* SRC_IO_TNS_FILE_FORMAT_H_ */
