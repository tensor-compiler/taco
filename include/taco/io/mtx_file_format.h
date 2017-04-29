#ifndef IO_MTX_FILE_FORMAT_H
#define IO_MTX_FILE_FORMAT_H

#include <fstream>
#include <vector>

namespace taco {
class TensorBase;

namespace io {
namespace mtx {

void writeFile(std::ofstream &mtxfile, std::string name,
               const std::vector<int> dimensions, int nnzero);

/// Read an mtx matrix from a file
TensorBase read(std::string filename, std::string name="");

/// Read an mtx matrix from a stream
TensorBase read(std::istream& stream, std::string name="");

/// Write an mtx matrix to a file
void write(std::string filename, const TensorBase& tensor);

/// write an mtx matrix to a stream
void write(std::ostream& stream, const TensorBase& tensor);

}}}

#endif
