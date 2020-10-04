#ifndef TACO_UTIL_FILES_H
#define TACO_UTIL_FILES_H

#include <string>
#include <fstream>

namespace taco {
namespace util {

std::string sanitizePath(std::string path);

void openStream(std::fstream& stream, std::string path, std::fstream::openmode mode);

}}
#endif
