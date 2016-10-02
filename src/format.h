#ifndef TACO_FORMAT_H
#define TACO_FORMAT_H

#include <string>
#include <memory>
#include <vector>
#include <ostream>

#include "tree.h"

namespace taco {

class Format {
public:
  Format();
  Format(std::string format);

  const std::vector<Level>& getLevels() const {return levels;}

  friend std::ostream &operator<<(std::ostream&, const Format&);

private:
  // The levels of the storage forest described by this format.
  std::vector<Level> levels;
};

}
#endif
