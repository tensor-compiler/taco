#include "taco/util/name_generator.h"

#include <atomic>
#include <sstream>

using namespace std;

namespace taco {
namespace util {

atomic<int> uniqueNameCounter;

static inline int uniqueCount() {
  return uniqueNameCounter++;
}

string uniqueName(char prefix) {
  return prefix + to_string(uniqueCount());
}

string uniqueName(const string& prefix) {
  return prefix + to_string(uniqueCount());
}


// class NameGenerator
NameGenerator::NameGenerator() {
}

NameGenerator::NameGenerator(std::vector<std::string> reserved) {
  for (auto& str : reserved) {
    nameCounters.insert({str,0});
  }
}

std::string NameGenerator::getUniqueName(std::string name) {
  stringstream os;
  os << name;
  if (nameCounters.count(name) > 0) {
    os << nameCounters[name]++;
  } else {
     nameCounters[name] = 0;
  }
  return os.str();
}

}}
