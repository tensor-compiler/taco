#ifndef TACO_FORMAT_H
#define TACO_FORMAT_H

#include <string>
#include <memory>
#include <vector>
#include <ostream>
#include "taco/type.h"

namespace taco {

class ModeType;
class ModeTypePack;
class ModeTypeImpl;


/// A Format describes the data layout of a tensor, and the sparse index data
/// structures that describe locations of non-zero tensor components.
class Format {
public:
  /// Create a format for a 0-order tensor (a scalar).
  Format();

  /// Create a format for a 1-order tensor (a vector).
  Format(const ModeType modeType);

  Format(const std::initializer_list<ModeTypePack>& modeTypePacks);
  
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
std::ostream& operator<<(std::ostream&, const Format&);



/// The type of a mode defines how it is stored.  For example, a mode may be
/// stored as a dense array, a compressed sparse representation, or a hash map.
/// New mode types can be defined by extending ModeTypeImpl.
class ModeType {
public:
  /// Aliases for predefined mode types
  static ModeType dense;       /// e.g., first mode in CSR
  static ModeType compressed;  /// e.g., second mode in CSR

  static ModeType sparse;      /// alias for compressed
  static ModeType Dense;       /// alias for dense
  static ModeType Compressed;  /// alias for compressed
  static ModeType Sparse;      /// alias for compressed

  /// Properties of a mode type
  enum Property {
    FULL, NOT_FULL, ORDERED, NOT_ORDERED, UNIQUE, NOT_UNIQUE, BRANCHLESS,
    NOT_BRANCHLESS, COMPACT, NOT_COMPACT
  };

  /// Instantiates an undefined mode type
  ModeType();

  /// Instantiates a new mode type
  ModeType(const std::shared_ptr<ModeTypeImpl> impl);

  /// Instantiates a variant of the mode type with differently configured
  /// properties
  ModeType operator()(const std::vector<Property>& properties = {});

  /// Returns true if mode type is defined, false otherwise. An undefined mode
  /// type can be used to indicate a mode whose format is not (yet) known.
  bool defined() const;

  /// Returns string identifying mode type. The format name should not reflect
  /// property configurations; mode types with differently configured properties
  /// should return the same name.
  std::string getName() const;

  /// Returns true if a mode type has a specific property, false otherwise
  bool isFull() const;
  bool isOrdered() const;
  bool isUnique() const;
  bool isBranchless() const;
  bool isCompact() const;

  /// Returns true if a mode type has a specific capability, false otherwise
  bool hasCoordValIter() const;
  bool hasCoordPosIter() const;
  bool hasLocate() const;
  bool hasInsert() const;
  bool hasAppend() const;

private:
  std::shared_ptr<const ModeTypeImpl> impl;

  friend class ModePack;
  friend class Iterator;
};

bool operator==(const ModeType&, const ModeType&);
bool operator!=(const ModeType&, const ModeType&);
std::ostream& operator<<(std::ostream&, const ModeType&);


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

bool operator==(const ModeTypePack&, const ModeTypePack&);
bool operator!=(const ModeTypePack&, const ModeTypePack&);
std::ostream& operator<<(std::ostream&, const ModeTypePack&);


/// Predefined formats
/// @{
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
/// @}

/// True if all modes are dense.
bool isDense(const Format&);

}
#endif
