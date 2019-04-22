#ifndef TACO_FORMAT_H
#define TACO_FORMAT_H

#include <string>
#include <memory>
#include <vector>
#include <ostream>
#include "taco/type.h"

namespace taco {

class ModeFormat;
class ModeFormatPack;
class ModeFormatImpl;


/// A Format describes the data layout of a tensor, and the sparse index data
/// structures that describe locations of non-zero tensor components.
class Format {
public:
  /// Create a format for a 0-order tensor (a scalar).
  Format();

  /// Create a format for a 1-order tensor (a vector).
  Format(const ModeFormat modeFormat);

  /// Create a tensor format whose modes have the given mode storage formats.
  /// The format of mode i is specified by modeFormats[i]. Mode i is stored in
  /// position i.
  Format(const std::vector<ModeFormat>& modeFormats);

  /// Create a tensor format where the modes have the given mode storage formats
  /// and modes are stored in the given sequence. The format of the mode stored
  /// in position i is specified by the i-th element of modeFormatPacks
  /// linearized. The mode stored in position i is specified by modeOrdering[i].
  Format(const std::vector<ModeFormat>& modeFormats,
         const std::vector<int>& modeOrdering);

  /// Create a tensor format where the modes have the given mode storage formats,
  /// are stored in the given sequence, and where contiguous modes share mode packs
  /// as specified by packBounds.
  /// The packBounds vector encodes the bounds between different packs For example,
  /// if a dense 4 tensor is stored in two packs, each containing two modes, then
  /// the bound between packs would be 2 (modes 0 and 1 in the first pack, and
  /// modes 2 and 3 in the second):
  ///
  /// Format({Dense, Dense, Dense, Dense}, {0,1,2,3}, {2});
  Format(const std::vector<ModeFormat>& modeFormats,
         const std::vector<int>& modeOrdering,
         const std::vector<int>& packBounds);

  /// Create a blocked tensor format. Blocks must all have the same dimensionality.
  ///
  /// Blocked formats create a level of abstraction in which each dimension of a
  /// tensor is divided into multiple modes. For each dimension, there must be
  /// exactly one block with non-fixed size.
  /// For example, to create a sparse matrix with dense 16x16 inner blocks:
  ///
  /// Format({{Dense, Sparse}, {Dense(16), Dense(16)}});
  Format(const std::vector<std::vector<ModeFormat>>& modeFormatBlocks);

  Format(const std::vector<std::vector<ModeFormat>>& modeFormatBlocks,
         const std::vector<int>& modeOrdering);

  Format(const std::vector<std::vector<ModeFormat>>& modeFormatBlocks,
         const std::vector<int>& modeOrdering,
         const std::vector<int>& packBounds);

  /// Returns the number of modes in the format.
  int getOrder() const;

  /// Get the storage types of the modes. The type of the mode stored in
  /// position i is specifed by element i of the returned vector.
  const std::vector<ModeFormat> getModeFormats() const;

  /// Get the storage formats of the modes, with modes that share the same
  /// physical storage grouped together.
  const std::vector<ModeFormatPack>& getModeFormatPacks() const;

  /// Get the ordering in which the modes are stored. The mode stored in
  /// position i is specifed by element i of the returned vector.
  const std::vector<int>& getModeOrdering() const;

  /// Gets the types of the coordinate arrays for each level
  const std::vector<std::vector<Datatype>>& getLevelArrayTypes() const;

  /// Gets the type of the position array for level i
  Datatype getCoordinateTypePos(size_t level) const;

  /// Gets the type of the idx array for level i
  Datatype getCoordinateTypeIdx(size_t level) const;

  /// Sets the types of the coordinate arrays for each level
  void setLevelArrayTypes(std::vector<std::vector<Datatype>> levelArrayTypes);

  /// Return true if Format represents a blocked tensor format
  bool isBlocked();

  /// Gets the number of nested block levels in the format
  int numberOfBlocks();

  /// Gets all the block fixed-sizes.
  ///
  /// A zero size is given for the free block dimension.
  /// Each tensor dimension contains exactly one free block
  /// size, which is determined at tensor instantiation.
  std::vector<std::vector<int>> getBlockSizes();

  /// For each dimension in the format, returns the block which contains
  /// the dimension's free size.
  std::vector<int> getDimensionFreeSizeBlock();

private:
  std::vector<ModeFormatPack> modeFormatPacks;
  std::vector<int> modeOrdering;
  std::vector<std::vector<Datatype>> levelArrayTypes;

  std::vector<ModeFormat> blockInit(
    const std::vector<std::vector<ModeFormat>>& modeFormatBlocks);
  bool blocked = false;
  int numBlocks;
  int numDims;
  std::vector<int> freeSizeBlock;
  std::vector<std::vector<int>> blockSizes;
};

bool operator==(const Format&, const Format&);
bool operator!=(const Format&, const Format&);
std::ostream& operator<<(std::ostream&, const Format&);


/// The type of a mode defines how it is stored.  For example, a mode may be
/// stored as a dense array, a compressed sparse representation, or a hash map.
/// New mode formats can be defined by extending ModeTypeImpl.
class ModeFormat {
public:
  /// Aliases for predefined mode formats
  static ModeFormat dense;       /// e.g., first mode in CSR
  static ModeFormat compressed;  /// e.g., second mode in CSR
  static ModeFormat singleton;   /// e.g., second mode in COO

  static ModeFormat sparse;      /// alias for compressed
  static ModeFormat Dense;       /// alias for dense
  static ModeFormat Compressed;  /// alias for compressed
  static ModeFormat Sparse;      /// alias for compressed
  static ModeFormat Singleton;   /// alias for singleton

  /// Properties of a mode format
  enum Property {
    FULL, NOT_FULL, ORDERED, NOT_ORDERED, UNIQUE, NOT_UNIQUE, BRANCHLESS,
    NOT_BRANCHLESS, COMPACT, NOT_COMPACT
  };

  /// Instantiates an undefined mode format
  ModeFormat();

  /// Instantiates a new mode format
  ModeFormat(const std::shared_ptr<ModeFormatImpl> impl);

  /// Instantiates a variant of the mode format with a differently configured
  /// property
  ModeFormat operator()(Property property) const;

  /// Instantiates a variant of the mode format with differently configured
  /// properties
  ModeFormat operator()(const std::vector<Property>& properties = {}) const;

  /// Instantiates a variant of the mode format with the given fixed size.
  ModeFormat operator()(const int size) const;

  /// Returns string identifying mode format. The format name should not reflect
  /// property configurations; mode formats with differently configured properties
  /// should return the same name.
  std::string getName() const;

  /// Returns true if the mode format has the given properties.
  bool hasProperties(const std::vector<Property>& properties) const;

  /// Returns true if a mode format has a specific property, false otherwise
  bool isFull() const;
  bool isOrdered() const;
  bool isUnique() const;
  bool isBranchless() const;
  bool isCompact() const;

  /// Returns true if a mode format has a specific capability, false otherwise
  bool hasCoordValIter() const;
  bool hasCoordPosIter() const;
  bool hasLocate() const;
  bool hasInsert() const;
  bool hasAppend() const;

  bool hasFixedSize() const;
  int size() const;

  /// Returns true if mode format is defined, false otherwise. An undefined mode
  /// type can be used to indicate a mode whose format is not (yet) known.
  bool defined() const;

  friend bool operator==(const ModeFormat&, const ModeFormat&);
  friend bool operator!=(const ModeFormat&, const ModeFormat&);
  friend std::ostream& operator<<(std::ostream&, const ModeFormat&);

private:
  std::shared_ptr<const ModeFormatImpl> impl;

  bool sizeFixed = false;
  int blockSize;

  friend class ModePack;
  friend class Iterator;
};


class ModeFormatPack {
public:
  ModeFormatPack(const std::vector<ModeFormat> modeFormats);
  ModeFormatPack(const std::initializer_list<ModeFormat> modeFormats);
  ModeFormatPack(const ModeFormat modeFormat);

  /// Get the storage types of the modes. The type of the mode stored in
  /// position i is specifed by element i of the returned vector.
  const std::vector<ModeFormat>& getModeFormats() const;

private:
  std::vector<ModeFormat> modeFormats;
};

bool operator==(const ModeFormatPack&, const ModeFormatPack&);
bool operator!=(const ModeFormatPack&, const ModeFormatPack&);
std::ostream& operator<<(std::ostream&, const ModeFormatPack&);


/// Predefined formats
/// @{
extern const ModeFormat Dense;
extern const ModeFormat Compressed;
extern const ModeFormat Sparse;
extern const ModeFormat Singleton;

extern const ModeFormat dense;
extern const ModeFormat compressed;
extern const ModeFormat sparse;
extern const ModeFormat singleton;

extern const Format CSR;
extern const Format CSC;
extern const Format DCSR;
extern const Format DCSC;

const Format COO(int order, bool isUnique = true, bool isOrdered = true, 
                 bool isAoS = false, const std::vector<int>& modeOrdering = {});
/// @}

/// True if all modes are dense.
bool isDense(const Format&);

}
#endif
