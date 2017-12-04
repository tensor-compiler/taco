#ifndef IO_H
#define IO_H

#include <string>
#include <fstream>

namespace taco {
namespace io {

std::string sanitizePath(std::string path);

std::fstream openStream(std::string path, std::ios_base::openmode mode);

}}
#endif
