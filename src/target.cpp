#include <string>
#include <map>
#include <vector>

#include "taco/target.h"

using namespace std;

namespace taco {

namespace {
map<string, Target::Arch> archMap = {{"c99", Target::C99},
                                      {"x86", Target::X86}};

map<string, Target::OS> osMap = {{"unknown", Target::OSUnknown},
                                  {"linux", Target::Linux},
                                  {"macos", Target::MacOS},
                                  {"windows", Target::Windows}};
  
bool parseTargetString(Target& target, string target_string) {
  string rest = target_string;
  vector<string> tokens;
  auto current_pos = rest.find('-');

  while (current_pos != string::npos) {
    tokens.push_back(rest.substr(0, current_pos));
    rest = rest.substr(current_pos+1);
  }
  
  // now parse the tokens
  taco_uassert(tokens.size() >= 2) <<
      "Invalid target string: " << target_string;
  
  // first must be architecture
  if (archMap.count(tokens[0]) == 0) {
    return false;
  }
  target.arch = archMap[tokens[0]];
  
  // next must be os
  if (osMap.count(tokens[1]) == 0) {
    return false;
  }
  target.os = osMap[tokens[1]];
  
  return true;
}

} // anonymous namespace

Target::Target(const std::string &s) {
  parseTargetString(*this, s);
}



bool Target::validateTargetString(const string &s) {
  string::size_type arch_end = string::npos;
  string::size_type os_end = string::npos;
  
  // locate arch
  for (auto res : archMap) {
    if (s.find(res.first) != string::npos) {
      arch_end = s.find(res.first);
    }
  }
  
  // locate os
  for (auto res : osMap) {
    if (s.find(res.first, arch_end+1) != string::npos) {
      os_end = s.find(res.first, arch_end+1);
    }
  }
  
  return (arch_end != string::npos) && (os_end != string::npos);
}

Target getTargetFromEnvironment() {
  return Target(Target::Arch::C99, Target::OS::MacOS);
}
} // namespace taco
