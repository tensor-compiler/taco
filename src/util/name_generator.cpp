#include "name_generator.h"

#include <atomic>

using namespace std;

namespace tacit {
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

}}
