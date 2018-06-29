#ifndef TACO_FORMAT_H
#define TACO_FORMAT_H

#include <string>
#include <memory>
#include <vector>
#include <ostream>
#include "taco/type.h"

#include "taco/storage/mode_type.h"
#include "taco/storage/dense_mode_type.h"
#include "taco/storage/compressed_mode_type.h"

namespace taco {

class ModeTypePack {
public:
  ModeTypePack(const std::vector<ModeType> modeTypes);
  ModeTypePack(const std::initializer_list<ModeType> modeTypes);
  ModeTypePack(const ModeType modeType);

  /// Get the storage types of the modes. The type of the mode stored in
  /// position i is specifed by element i of the returned vector.
  const std::vector<ModeType>& getModeTypes() const;

private:
  std::vector<ModeType> modeTypes;
};


/// A Format describes the data layout of a tensor, and the sparse index data
/// structures that describe locations of non-zero tensor components.
class Format {
public:
  /// Create a format for a 0-order tensor (a scalar).
  Format();

  /// Create a format for a 1-order tensor (a vector).
  Format(const ModeType modeType);

  /// Create a tensor format whose modes have the given storage types. The type
  /// of mode i is specified by modeTypes[i]. Mode i is stored in position i.
  Format(const std::vector<ModeTypePack>& modeTypePacks);

  /// Create a tensor format where the modes have the given storage types and
  /// modes are stored in the given sequence. The type of the mode stored in
  /// position i is specified by the i-th element of modeTypePacks linearized. 
  /// The mode stored in position i is specified by modeOrdering[i].
  Format(const std::vector<ModeTypePack>& modeTypePacks,
         const std::vector<size_t>& modeOrdering);

  /// Returns the number of modes in the format.
  size_t getOrder() const;

  /// Get the storage types of the modes. The type of the mode stored in
  /// position i is specifed by element i of the returned vector.
  const std::vector<ModeType> getModeTypes() const;

  /// Get the storage types of the modes, with modes that share the same 
  /// physical storage grouped together.
  const std::vector<ModeTypePack>& getModeTypePacks() const;

  /// Get the ordering in which the modes are stored. The mode stored in
  /// position i is specifed by element i of the returned vector.
  const std::vector<size_t>& getModeOrdering() const;

  /// Gets the types of the coordinate arrays for each level
  const std::vector<std::vector<Datatype>>& getLevelArrayTypes() const;

  /// Gets the type of the position array for level i
  Datatype getCoordinateTypePos(int level) const;

  /// Gets the type of the idx array for level i
  Datatype getCoordinateTypeIdx(int level) const;

  /// Sets the types of the coordinate arrays for each level
  void setLevelArrayTypes(std::vector<std::vector<Datatype>> levelArrayTypes);

private:
  std::vector<ModeTypePack> modeTypePacks;
  std::vector<size_t> modeOrdering;
  std::vector<std::vector<Datatype>> levelArrayTypes;
};

bool operator==(const Format&, const Format&);
bool operator!=(const Format&, const Format&);
bool operator==(const ModeTypePack&, const ModeTypePack&);
bool operator!=(const ModeTypePack&, const ModeTypePack&);

std::ostream& operator<<(std::ostream&, const Format&);
std::ostream& operator<<(std::ostream&, const ModeTypePack&);


// Predefined formats
extern const ModeType Dense;
extern const ModeType Compressed;
extern const ModeType Sparse;

extern const ModeType dense;
extern const ModeType compressed;
extern const ModeType sparse;

extern const Format CSR;
extern const Format CSC;
extern const Format DCSR;
extern const Format DCSC;

/// True if all modes are Dense
bool isDense(const Format&);

}
#endif
