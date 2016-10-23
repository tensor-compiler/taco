#ifndef TACO_FORMAT_H
#define TACO_FORMAT_H

#include <string>
#include <memory>
#include <vector>
#include <ostream>

namespace taco {
class Level;

enum LevelType {
  Dense,
  Sparse,
  Fixed,
//  Repeated,
//  Replicated
};

class Format {
public:
  Format();
  Format(std::vector<Level> levels);
  Format(std::vector<LevelType> levelTypes, std::vector<size_t> dimensionOrder);
  Format(std::vector<LevelType> levelTypes);

  const std::vector<Level>& getLevels() const {return levels;}

private:
  // The levels of the storage forest described by this format.
  std::vector<Level> levels;
};

std::ostream &operator<<(std::ostream&, const Format&);


class Level {
public:
  Level(size_t dimension, LevelType type) : dimension(dimension), type(type) {}

  LevelType getType() const {
    return type;
  }

  size_t getDimension() const {
    return dimension;
  }

private:
  size_t dimension;  // The tensor dimension described by the format level
  LevelType type;
};

std::ostream& operator<<(std::ostream&, const LevelType&);
std::ostream& operator<<(std::ostream&, const Level&);

}
#endif
