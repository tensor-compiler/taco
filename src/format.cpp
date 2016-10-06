#include "format.h"

#include <iostream>

#include "tree.h"

using namespace std;

namespace taco {

// class Format
Format::Format() {
}

Format::Format(std::string format) {
  for (size_t i=0; i < format.size(); ++i) {
    switch (format[i]) {
      case 'd':
        levels.push_back(Level::Dense);
        break;
      case 's':
        levels.push_back(Level::Sparse);
        break;
      case 'f':
        not_supported_yet;
        break;
      case 'r':
        not_supported_yet;
        break;
      default:
        uerror << "Format character not recognized: " << format[i];
    }
  }
  levels.push_back(Level::Values);
}

std::ostream &operator<<(std::ostream& os, const Format& format) {
  for (auto& level : format.getLevels()) {
    switch (level.type) {
      case Level::Dense:
        os << "d";
        break;
      case Level::Sparse:
        os << "s";
        break;
      case Level::Values:
        break;
    }
  }
  return os;
}

}
