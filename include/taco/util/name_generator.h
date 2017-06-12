#ifndef TACO_UTIL_NAME_GENERATOR_H
#define TACO_UTIL_NAME_GENERATOR_H

#include <string>
#include <map>
#include <vector>

namespace taco {
namespace util {

std::string uniqueName(char prefix);
std::string uniqueName(const std::string& prefix);

class NameGenerator {
public:
  NameGenerator();
  NameGenerator(std::vector<std::string> reserved);

  std::string getUniqueName(std::string name);

private:
  std::map<std::string, int> nameCounters;
};

}}
#endif
