#ifndef IO_TNS_FILE_FORMAT_H
#define IO_TNS_FILE_FORMAT_H

#include <fstream>
#include <vector>

namespace taco {
class TensorBase;

namespace io {
namespace tns {

void writeFile(std::ofstream &mtxfile, std::string name,
               const TensorBase *tensor);

/// Read an tns tensor from a file
TensorBase read(std::string filename, std::string name="");

/// Read a tns tensor from a stream
TensorBase read(std::istream& stream, std::string name="");

/// Write a tns tensor to a file
void write(std::string filename, const TensorBase& tensor);

/// Write a tns tensor to a stream
void write(std::ostream& stream, const TensorBase& tensor);

}}}

#endif
