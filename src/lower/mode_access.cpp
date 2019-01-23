#include "lower/mode_access.h"

namespace taco {

ModeAccess::ModeAccess(Access access, int mode) : access(access), mode(mode){
}

Access ModeAccess::getAccess() const {
  return access;
}

size_t ModeAccess::getMode() const {
  return mode;
}

bool operator==(const ModeAccess& a, const ModeAccess& b) {
  return a.getAccess() == b.getAccess() && a.getMode() == b.getMode();
}

bool operator<(const ModeAccess& a, const ModeAccess& b) {
  if (a.getAccess() == b.getAccess()) {
    return a.getMode() < b.getMode();
  }
  return a.getAccess() < b.getAccess();
}

std::ostream &operator<<(std::ostream &os, const ModeAccess & modeAccess) {
  return os << modeAccess.getAccess().getTensorVar().getName()
            << "(" << modeAccess.getMode() << ")";
}

}
