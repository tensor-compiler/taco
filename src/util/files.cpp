#include "taco/util/files.h"

#include "taco/error.h"

#include <iostream>
#include <fstream>
#include <cstdlib>

using namespace std;

namespace taco {
namespace util {

std::string sanitizePath(std::string path) {
  if (path[0] == '~') {
    path = path.replace(0, 1, std::getenv("HOME"));
  }

  return path;
}

void openStream(std::fstream& stream, std::string path, fstream::openmode mode) {
  stream.open(sanitizePath(path), mode);
  taco_uassert(stream.is_open()) << "Error opening file: " << path;
}

}}
