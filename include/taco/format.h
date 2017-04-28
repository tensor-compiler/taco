#ifndef TACO_FORMAT_H
#define TACO_FORMAT_H

#include <string>
#include <memory>
#include <vector>
#include <ostream>

namespace taco {
class Level;

enum LevelType {
  Dense,      // e.g. first  dimension in CSR
  Sparse,     // e.g. second dimension in CSR
  Fixed       // e.g. second dimension in ELL
};

class Format {
public:
  /// Create a format for a tensor with no dimensions
  Format();

  /// Create a tensor format where the levels have the given storage types.
  /// The levels are ordered the same way as the dimensions.
  Format(const std::vector<LevelType>& levelTypes);

  /// Create a tensor format where the levels have the given storage types and
  /// dimension order.
  Format(const std::vector<LevelType>& levelTypes,
         const std::vector<size_t>& dimensionOrder);

  /// Get the tensor storage levels.
  const std::vector<Level>& getLevels() const {return levels;}

private:
  std::vector<Level> levels;
};

bool operator==(const Format&, const Format&);
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


// Predefined formats
extern const Format DVEC;
extern const Format SVEC;

extern const Format DMAT;
extern const Format CSR;
extern const Format CSC;
extern const Format ELL;

}
#endif
