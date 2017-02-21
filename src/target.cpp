#include <string>
#include <map>

#include "target.h"

using namespace std;

namespace taco {

namespace {
map<string, Target::Arch> arch_map = {{"c99", Target::C99},
                                      {"x86", Target::X86}};

map<string, Target::OS> os_map = {{"unknown", Target::OSUnknown},
                                  {"linux", Target::Linux},
                                  {"macos", Target::MacOS},
                                  {"windows", Target::Windows}};

} // anonymous namespace


bool Target::validate_target_string(const string &s) {
  string::size_type arch_end = string::npos;
  string::size_type os_end = string::npos;
  
  // locate arch
  for (auto res : arch_map) {
    if (s.find(res.first) != string::npos) {
      arch_end = s.find(res.first);
    }
  }
  
  // locate os
  for (auto res : os_map) {
    if (s.find(res.first, arch_end+1) != string::npos) {
      os_end = s.find(res.first, arch_end+1);
    }
  }
  
  return (arch_end != string::npos) && (os_end != string::npos);
}

} // namespace taco
