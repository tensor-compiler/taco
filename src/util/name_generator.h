#ifndef TACIT_UTIL_NAME_GENERATOR_H
#define TACIT_UTIL_NAME_GENERATOR_H

#include <string>

namespace tacit {
namespace util {

std::string uniqueName(char prefix);
std::string uniqueName(const std::string& prefix);

}}
#endif
