#include "taco/io/dns_file_format.h"

#include <iostream>
#include <fstream>

#include "taco/tensor.h"
#include "taco/error.h"

using namespace std;

namespace taco {
namespace io {
namespace dns {

TensorBase read(std::string filename, const Format& format, bool pack) {
  std::ifstream file;
  file.open(filename);
  taco_uassert(file.is_open()) << "Error opening file: " << filename;
  TensorBase tensor = read(file, format, pack);
  file.close();
  return tensor;
}

TensorBase read(std::istream& stream, const Format& format, bool pack) {
  taco_not_supported_yet;
  return TensorBase();
}

void write(std::string filename, const TensorBase& tensor) {
  std::ofstream file;
  file.open(filename);
  taco_uassert(file.is_open()) << "Error opening file: " << filename;
  write(file, tensor);
  file.close();
}

void write(std::ostream& stream, const TensorBase& tensor) {
}

}}}
