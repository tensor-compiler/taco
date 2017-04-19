#ifndef SRC_IO_TNS_FILE_FORMAT_H_
#define SRC_IO_TNS_FILE_FORMAT_H_

#include <fstream>
#include <vector>

namespace taco {
class TensorBase;

namespace io {
namespace tns {

void readFile(std::ifstream &tnsfile, std::vector<int> &dims, 
              TensorBase* tensor);

//void writeFile(std::ofstream &mtxfile, std::string name, TensorBase* tensor);

}}}

#endif /* SRC_IO_MTX_FILE_FORMAT_H_ */
