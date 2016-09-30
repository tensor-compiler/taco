#ifndef TACO_FORMAT_H
#define TACO_FORMAT_H

#include <string>
#include <memory>
#include <vector>
#include <ostream>

namespace taco {

class TreeLevel;
struct Level;

class Format {
public:
  Format();
  Format(std::string format);

  const std::vector<std::shared_ptr<Level>>& getLevels() const {return levels;}

  friend std::ostream &operator<<(std::ostream&, const Format&);

private:
  // The levels of the storage forest described by this format.
  std::vector<std::shared_ptr<Level>> levels;
};

}
#endif
