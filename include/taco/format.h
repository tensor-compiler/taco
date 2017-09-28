#ifndef TACO_FORMAT_H
#define TACO_FORMAT_H

#include <string>
#include <memory>
#include <vector>
#include <ostream>

namespace taco {

enum ModeType {
  Dense,   // e.g. first  mode in CSR
  Sparse,  // e.g. second mode in CSR
  Fixed    // e.g. second mode in ELL
};

class Format {
public:
  /// Create a format for a 0-order tensor.
  Format();

  /// Create a tensor format whose modes have the same storage type.
  Format(const ModeType& modeType);

  /// Create a tensor format where the modes have the given storage types. The
  /// type of mode i is specified by modeTypes[i]. Mode i will be stored in
  /// position i.
  Format(const std::vector<ModeType>& modeTypes);

  /// Create a tensor format where the modes have the given storage types and
  /// modes are stored in the given sequence. The type of the mode stored in
  /// position i is specified by modeTypes[i]. The mode stored in position i is
  /// specified by modeOrdering[i].
  Format(const std::vector<ModeType>& modeTypes,
         const std::vector<size_t>& modeOrdering);

  /// Returns the number of modes in the format.
  size_t getOrder() const;

  /// Get the storage types of the modes. The type of the mode stored in
  /// position i is specifed by element i of the returned vector.
  const std::vector<ModeType>& getModeTypes() const;

  /// Get the ordering in which the modes are stored. The mode stored in
  /// position i is specifed by element i of the returned vector.
  const std::vector<size_t>& getModeOrdering() const;

private:
  std::vector<ModeType> modeTypes;
  std::vector<size_t>   modeOrdering;
};

bool operator==(const Format&, const Format&);
bool operator!=(const Format&, const Format&);

std::ostream& operator<<(std::ostream&, const Format&);
std::ostream& operator<<(std::ostream&, const ModeType&);


// Predefined formats
extern const Format CSR;
extern const Format CSC;
extern const Format DCSR;
extern const Format DCSC;

/// True if all modes are Dense
bool isDense(const Format&);

}
#endif
