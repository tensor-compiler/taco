#ifndef TACO_UTIL_PRINTABLE_H
#define TACO_UTIL_PRINTABLE_H

#include <ostream>

namespace taco {
namespace util {

class Printable {
public:
  virtual void print(std::ostream &os) const = 0;
  virtual ~Printable() {};
};

inline std::ostream &operator<<(std::ostream &os, const Printable &printable) {
  printable.print(os);
  return os;
}

}}
#endif
