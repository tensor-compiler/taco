#ifndef TACO_FORMAT_H
#define TACO_FORMAT_H

#include <string>
#include <memory>
#include <vector>
#include <ostream>

namespace taco {
class Level;

enum DimensionType {
  Dense,      // e.g. first  dimension in CSR
  Sparse,     // e.g. second dimension in CSR
  Fixed       // e.g. second dimension in ELL
};

class Format {
public:
  /// Create a format for a tensor with no dimensions
  Format();

  /// Create a tensor format that can be used with any tensor and whose
  /// dimensions have the same storage type.
  Format(const DimensionType& dimensionType);

  /// Create a tensor format where the dimensions have the given storage types.
  /// The dimensions are ordered from first to last.
  Format(const std::vector<DimensionType>& dimensionTypes);

  /// Create a tensor format where the dimensions have the given storage types and
  /// dimension order.
  Format(const std::vector<DimensionType>& dimensionTypes,
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
  Level(size_t dimension, DimensionType type) : dimension(dimension), type(type) {}

  DimensionType getType() const {
    return type;
  }

  size_t getDimension() const {
    return dimension;
  }

private:
  size_t dimension;  // The tensor dimension described by the format level
  DimensionType type;
};

std::ostream& operator<<(std::ostream&, const DimensionType&);
std::ostream& operator<<(std::ostream&, const Level&);


// Predefined formats
extern const Format CSR;
extern const Format CSC;
extern const Format DCSR;
extern const Format DCSC;

}
#endif
