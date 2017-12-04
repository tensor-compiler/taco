#include "io/io_util.h"

#include "taco/error.h"

#include <iostream>
#include <cstdlib>

using namespace std;

namespace taco {
namespace io {

std::string sanitizePath(std::string path) {
  if (path[0] == '~') {
    path = path.replace(0, 1, std::getenv("HOME"));
    std::cout << path << std::endl;
  }

  return path;
}

std::fstream openStream(std::string path, ios_base::openmode mode) {
  std::fstream file;
  file.open(sanitizePath(path), mode);
  taco_uassert(file.is_open()) << "Error opening file: " << path;
  return file;
}

}}
