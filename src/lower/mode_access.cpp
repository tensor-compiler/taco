#include "lower/mode_access.h"

namespace taco {

ModeAccess::ModeAccess(Access access, int mode) : access(access), mode(mode){
}

Access ModeAccess::getAccess() const {
  return access;
}

size_t ModeAccess::getModePos() const {
  return mode;
}

bool operator==(const ModeAccess& a, const ModeAccess& b) {
  return a.getAccess() == b.getAccess() && a.getModePos() == b.getModePos();
}

bool operator<(const ModeAccess& a, const ModeAccess& b) {
  if (a.getAccess() == b.getAccess()) {
    return a.getModePos() < b.getModePos();
  }
  return a.getAccess() < b.getAccess();
}

std::ostream &operator<<(std::ostream &os, const ModeAccess & modeAccess) {
  return os << modeAccess.getAccess().getTensorVar().getName()
            << "(" << modeAccess.getModePos() << ")";
}

}
