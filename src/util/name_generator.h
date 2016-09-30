#ifndef TAC_UTIL_NAME_GENERATOR_H
#define TAC_UTIL_NAME_GENERATOR_H

#include <string>

namespace tac {
namespace util {

std::string uniqueName(char prefix);
std::string uniqueName(const std::string& prefix);

}}
#endif
