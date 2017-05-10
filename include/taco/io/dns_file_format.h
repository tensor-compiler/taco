#ifndef IO_DNS_FILE_FORMAT_H
#define IO_DNS_FILE_FORMAT_H

#include <istream>
#include <ostream>
#include <string>

namespace taco {
class TensorBase;
class Format;
namespace io {
namespace dns {

/// Read a dns tensor from a file.
TensorBase read(std::string filename, const Format& format, bool pack = true);

/// Read a dns tensor from a stream.
TensorBase read(std::istream& stream, const Format& format, bool pack = true);

/// Write a dns tensor to a file.
void write(std::string filename, const TensorBase& tensor);

/// Write a dns tensor to a stream.
void write(std::ostream& stream, const TensorBase& tensor);

}}}

#endif
