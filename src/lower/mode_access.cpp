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

static bool accessEqual(const Access& a, const Access& b) {
  return a == b ||
         (a.getTensorVar() == b.getTensorVar() && a.getIndexVars() == b.getIndexVars());
}

bool operator==(const ModeAccess& a, const ModeAccess& b) {
  return accessEqual(a.getAccess(), b.getAccess()) && a.getModePos() == b.getModePos();
}

bool operator<(const ModeAccess& a, const ModeAccess& b) {

  // fast path for when access pointers are equal
  if(a.getAccess() == b.getAccess()) {
    return a.getModePos() < b.getModePos();
  }

  // First break on tensorVars
  if(a.getAccess().getTensorVar() != b.getAccess().getTensorVar()) {
    return a.getAccess().getTensorVar() < b.getAccess().getTensorVar();
  }

  // Then break on the indexVars used in the access
  std::vector<IndexVar> aVars = a.getAccess().getIndexVars();
  std::vector<IndexVar> bVars = b.getAccess().getIndexVars();

  if(aVars.size() != bVars.size()) {
    return aVars.size() < bVars.size();
  }

  for(size_t i = 0; i < aVars.size(); ++i) {
    if(aVars[i] != bVars[i]) {
      return aVars[i] < bVars[i];
    }
  }

  // Finally, break on the mode position
  return a.getModePos() < b.getModePos();
}

std::ostream &operator<<(std::ostream &os, const ModeAccess & modeAccess) {
  return os << modeAccess.getAccess().getTensorVar().getName()
            << "(" << modeAccess.getModePos() << ")";
}

}
