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

  /// Create a tensor format where the modes have the given storage types,
  /// ordered from first to last.
  Format(const std::vector<ModeType>& modeTypes);

  /// Create a tensor format where the modes have the given storage types and
  /// mode order.
  Format(const std::vector<ModeType>& modeTypes,
         const std::vector<int>& modeOrder);

  /// Returns the number of modes in the format.
  size_t getOrder() const;

  /// Get the storage types of the modes.
  const std::vector<ModeType>& getModeTypes() const;

  /// Get the storage order of the modes. The storage order is a permutation
  /// vector where location i contains the storage location of mode i.
  const std::vector<int>& getModeOrder() const;

private:
  std::vector<ModeType> modeTypes;
  std::vector<int>      modeOrder;
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

/// True if all dimensions are Dense
bool isDense(const Format&);

}
#endif
