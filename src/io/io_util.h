#ifndef IO_H
#define IO_H

#include <string>
#include <fstream>

namespace taco {
namespace io {

std::string sanitizePath(std::string path);

void openStream(std::fstream& stream, std::string path, std::fstream::openmode mode);

}}
#endif
