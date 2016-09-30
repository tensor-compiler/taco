#ifndef TAC_UTIL_PRINTABLE_H
#define TAC_UTIL_PRINTABLE_H

#include <ostream>

namespace tac {
namespace util {

class Printable {
public:
  virtual void print(std::ostream &os) const = 0;
};

inline std::ostream &operator<<(std::ostream &os, const Printable &printable) {
  printable.print(os);
  return os;
}

}}
#endif
