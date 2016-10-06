#ifndef TACO_UTIL_NAME_GENERATOR_H
#define TACO_UTIL_NAME_GENERATOR_H

#include <string>

namespace taco {
namespace util {

std::string uniqueName(char prefix);
std::string uniqueName(const std::string& prefix);

}}
#endif
