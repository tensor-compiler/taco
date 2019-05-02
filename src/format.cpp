#include "taco/format.h"

#include <iostream>
#include <climits>
#include <vector>
#include <typeinfo>
#include <initializer_list>

#include "taco/lower/mode_format_dense.h"
#include "taco/lower/mode_format_dense_old.h"
#include "taco/lower/mode_format_compressed.h"
#include "taco/lower/mode_format_singleton.h"

#include "taco/error.h"
#include "taco/util/strings.h"

using namespace std;

namespace taco {

Format::Format() {
}

Format::Format(const ModeFormat modeFormat) : modeFormatPacks({modeFormat}),
    modeOrdering({0}) {}

Format::Format(const std::vector<ModeFormat>& modeFormats) {
  int order = modeFormats.size();
  taco_uassert(order <= INT_MAX) << "Supports only INT_MAX modes";

  modeOrdering.resize(order);
  for (int i = 0; i < order; ++i) {
    modeOrdering[i] = i;
    modeFormatPacks.push_back(modeFormats[i]);
  }
}

Format::Format(const std::vector<ModeFormat>& modeFormats,
               const std::vector<int>& modeOrdering)
    : modeOrdering(modeOrdering) {
  int order = modeFormats.size();
  taco_uassert(order <= INT_MAX) << "Supports only INT_MAX modes";
  taco_uassert((size_t)order == modeOrdering.size()) <<
      "You must either provide a complete mode ordering or none";

  for (ModeFormat format : modeFormats) {
    modeFormatPacks.push_back(format);
  }
}

Format::Format(const std::vector<ModeFormat>& modeFormats,
               const std::vector<int>& modeOrdering,
               const std::vector<int>& packBounds)
    : modeOrdering(modeOrdering) {
  int order = modeFormats.size();
  taco_uassert(order <= INT_MAX) << "Supports only INT_MAX modes";
  taco_uassert((size_t)order == modeOrdering.size()) <<
      "You must either provide a complete mode ordering or none";

  // Check for correct packBounds format.
  int leftBound = 0;
  for (int rightBound : packBounds) {
    taco_uassert(rightBound < order) << "pack delimiters cannot exceed tensor order";
    taco_uassert(leftBound < rightBound) << "packs must contain at least one dimension";
    leftBound = rightBound;
  }

  int pack = 0;
  std::vector<ModeFormat> modeFormatPack;
  for (int i = 0; i < (int)order; ++i) {
    if (i == packBounds[pack]) {
      modeFormatPacks.push_back(modeFormatPack);
      modeFormatPack.clear();
      ++pack;
    }
    modeFormatPack.push_back(modeFormats[i]);
  }
  modeFormatPacks.push_back(modeFormatPack);
}

/// Extracts block metadata for format and returns vector of all
/// modeFormats in format.
std::vector<ModeFormat> Format::blockInit(
    const std::vector<std::vector<ModeFormat>>& modeFormatBlocks) {
  blocked = true;
  numBlocks = modeFormatBlocks.size();
  numDims = modeFormatBlocks[0].size();
  std::vector<ModeFormat> modeFormats(numBlocks * numDims);
  freeSizeBlock = std::vector<int>(numDims, -1);
  blockSizes = std::vector<std::vector<int>>(numBlocks);

  for (int block_i = 0; block_i < numBlocks; block_i++) {
    std::vector<ModeFormat> block = modeFormatBlocks[block_i];
    taco_uassert(block.size() == (size_t)numDims) <<
        "All blocks must have the same dimensionality";

    blockSizes[block_i] = std::vector<int>(numDims, 0);
    for (int dim_i = 0; dim_i < numDims; dim_i++) {
      ModeFormat format = block[dim_i];
      modeFormats[block_i * numDims + dim_i] = format;
      if (format.hasFixedSize()) {
        blockSizes[block_i][dim_i] = format.size();
      } else {
        taco_uassert(freeSizeBlock[dim_i] == -1) <<
            "Each dimension requires at most one free-size block.";
        freeSizeBlock[dim_i] = block_i;
      }
    }
  }
  for (int dim_i = 0; dim_i < numDims; dim_i++) {
    taco_uassert(freeSizeBlock[dim_i] != -1) <<
        "Each dimension requires at least one free-size block.";
  }
  return modeFormats;
}

Format::Format(const std::vector<std::vector<ModeFormat>>& modeFormatBlocks) {
  std::vector<ModeFormat> modeFormats = blockInit(modeFormatBlocks);
  int order = modeFormats.size();
  taco_uassert(order <= INT_MAX) << "Supports only INT_MAX modes";

  modeOrdering.resize(order);
  for (int i = 0; i < order; ++i) {
    modeOrdering[i] = i;
    modeFormatPacks.push_back(modeFormats[i]);
  }
}

Format::Format(const std::vector<std::vector<ModeFormat>>& modeFormatBlocks,
               const std::vector<int>& modeOrdering)
    : modeOrdering(modeOrdering) {
  std::vector<ModeFormat> modeFormats = blockInit(modeFormatBlocks);
  int order = modeFormats.size();
  taco_uassert(order <= INT_MAX) << "Supports only INT_MAX modes";
  taco_uassert((size_t)order == modeOrdering.size()) <<
      "You must either provide a complete mode ordering or none";

  for (ModeFormat format : modeFormats) {
    modeFormatPacks.push_back(format);
  }
}

Format::Format(const std::vector<std::vector<ModeFormat>>& modeFormatBlocks,
               const std::vector<int>& modeOrdering,
               const std::vector<int>& packBounds)
    : modeOrdering(modeOrdering) {
  std::vector<ModeFormat> modeFormats = blockInit(modeFormatBlocks);

  int order = modeFormats.size();
  taco_uassert(order <= INT_MAX) << "Supports only INT_MAX modes";
  taco_uassert((size_t)order == modeOrdering.size()) <<
      "You must either provide a complete mode ordering or none";
  
  // Check for correct packBounds format.
  int leftBound = 0;
  for (int rightBound : packBounds) {
    taco_uassert(rightBound < order) << "pack delimiters cannot exceed tensor order";
    taco_uassert(leftBound < rightBound) << "packs must contain at least one dimension";
    leftBound = rightBound;
  }

  int pack = 0;
  std::vector<ModeFormat> modeFormatPack;
  for (int i = 0; i < (int)order; ++i) {
    if (i == packBounds[pack]) {
      modeFormatPacks.push_back(modeFormatPack);
      modeFormatPack.clear();
      ++pack;
    }
    modeFormatPack.push_back(modeFormats[i]);
  }
  modeFormatPacks.push_back(modeFormatPack);
}

int Format::getOrder() const {
  return (int)getModeFormats().size();
}

const std::vector<ModeFormat> Format::getModeFormats() const {
  std::vector<ModeFormat> modeFormats;
  for (auto modeFormatPack : getModeFormatPacks()) {
    modeFormats.insert(modeFormats.end(),
                       modeFormatPack.getModeFormats().begin(),
                       modeFormatPack.getModeFormats().end());
  }
  return modeFormats;
}

const std::vector<ModeFormatPack>& Format::getModeFormatPacks() const {
  return this->modeFormatPacks;
}

const std::vector<int>& Format::getModeOrdering() const {
  return this->modeOrdering;
}

const std::vector<std::vector<Datatype>>& Format::getLevelArrayTypes() const {
  return this->levelArrayTypes;
}

Datatype Format::getCoordinateTypePos(size_t level) const {
  if (level >= levelArrayTypes.size()) {
    return Int32;
  }
  return levelArrayTypes[level][0];
}

Datatype Format::getCoordinateTypeIdx(size_t level) const {
  if (level >= levelArrayTypes.size()) {
    return Int32;
  }
  if (getModeFormats()[level].getName() == Dense.getName()) {
    return levelArrayTypes[level][0];
  }
  return levelArrayTypes[level][1];
}

void Format::setLevelArrayTypes(std::vector<std::vector<Datatype>> levelArrayTypes) {
  this->levelArrayTypes = levelArrayTypes;
}

bool Format::isBlocked() {
  return blocked;
}

int Format::numBlockLevels() {
  taco_uassert(isBlocked()) << "ModeFormat does not have fixed size.";
  return numBlocks;
}

std::vector<std::vector<int>> Format::getBlockSizes() {
  taco_uassert(isBlocked()) << "ModeFormat does not have fixed size.";
  return blockSizes;
}

std::vector<int> Format::getDimensionFreeSizeBlock() {
  taco_uassert(isBlocked()) << "ModeFormat does not have fixed size.";
  return freeSizeBlock;
}


bool operator==(const Format& a, const Format& b){
  const auto aModeTypePacks = a.getModeFormatPacks();
  const auto bModeTypePacks = b.getModeFormatPacks();
  const auto aModeOrdering = a.getModeOrdering();
  const auto bModeOrdering = b.getModeOrdering();
  
  if (aModeTypePacks.size() != bModeTypePacks.size() || 
      aModeOrdering.size() != bModeOrdering.size()) {
    return false;
  }
  for (size_t i = 0; i < aModeOrdering.size(); ++i) {
    if (aModeOrdering[i] != bModeOrdering[i]) {
      return false;
    }
  }
  for (size_t i = 0; i < aModeTypePacks.size(); i++) {
    if (aModeTypePacks[i] != bModeTypePacks[i]) {
      return false;
    }
  } 
  return true;
}

bool operator!=(const Format& a, const Format& b) {
  return !(a == b);
}

std::ostream &operator<<(std::ostream& os, const Format& format) {
  return os << "(" << util::join(format.getModeFormatPacks(), ",") << "; "
            << util::join(format.getModeOrdering(), ",") << ")";
}


// class ModeType
ModeFormat::ModeFormat() {
}

ModeFormat::ModeFormat(const std::shared_ptr<ModeFormatImpl> impl) : 
    impl(impl) {
}

ModeFormat ModeFormat::operator()(Property property) const {
  return defined() ? impl->copy({property}) : ModeFormat();
}

ModeFormat ModeFormat::operator()(
    const std::vector<Property>& properties) const {
  return defined() ? impl->copy(properties) : ModeFormat();
}

ModeFormat ModeFormat::operator()(const int size) const {
  return defined() ? impl->copy({SIZE_FIXED}, size) : ModeFormat();
}

std::string ModeFormat::getName() const {
  return defined() ? impl->name : "undefined";
}

bool ModeFormat::hasProperties(const std::vector<Property>& properties) const {
  for (auto& property : properties) {
    switch (property) {
      case FULL:
        if (!isFull()) {
          return false;
        }
        break;
      case ORDERED:
        if (!isOrdered()) {
          return false;
        }
        break;
      case UNIQUE:
        if (!isUnique()) {
          return false;
        }
        break;
      case BRANCHLESS:
        if (!isBranchless()) {
          return false;
        }
        break;
      case COMPACT:
        if (!isCompact()) {
          return false;
        }
        break;
      case SIZE_FIXED:
        if (!hasFixedSize()) {
          return false;
        }
        break;
      case NOT_FULL:
        if (isFull()) {
          return false;
        }
        break;
      case NOT_ORDERED:
        if (isOrdered()) {
          return false;
        }
        break;
      case NOT_UNIQUE:
        if (isUnique()) {
          return false;
        }
        break;
      case NOT_BRANCHLESS:
        if (isBranchless()) {
          return false;
        }
        break;
      case NOT_COMPACT:
        if (isCompact()) {
          return false;
        }
        break;
      case SIZE_NOT_FIXED:
        if (hasFixedSize()) {
          return false;
        }
        break;
    }
  }
  return true;
}

bool ModeFormat::isFull() const {
  taco_iassert(defined());
  return impl->isFull;
}

bool ModeFormat::isOrdered() const {
  taco_iassert(defined());
  return impl->isOrdered;
}

bool ModeFormat::isUnique() const {
  taco_iassert(defined());
  return impl->isUnique;
}

bool ModeFormat::isBranchless() const {
  taco_iassert(defined());
  return impl->isBranchless;
}

bool ModeFormat::isCompact() const {
  taco_iassert(defined());
  return impl->isCompact;
}

bool ModeFormat::hasCoordValIter() const {
  taco_iassert(defined());
  return impl->hasCoordValIter;
}

bool ModeFormat::hasCoordPosIter() const {
  taco_iassert(defined());
  return impl->hasCoordPosIter;
}

bool ModeFormat::hasLocate() const {
  taco_iassert(defined());
  return impl->hasLocate;
}

bool ModeFormat::hasInsert() const {
  taco_iassert(defined());
  return impl->hasInsert;
}

bool ModeFormat::hasAppend() const {
  taco_iassert(defined());
  return impl->hasAppend;
}

bool ModeFormat::hasFixedSize() const {
  taco_iassert(defined());
  return impl->hasFixedSize;
}

int ModeFormat::size() const {
  taco_iassert(defined());
  taco_uassert(hasFixedSize()) << "ModeFormat does not have fixed size.";
  return impl->size;
}

bool ModeFormat::defined() const {
  return impl != nullptr;
}

bool operator==(const ModeFormat& a, const ModeFormat& b) {
  return (a.defined() && b.defined() && (*a.impl == *b.impl));
}

bool operator!=(const ModeFormat& a, const ModeFormat& b) {
  return !(a == b);
}

std::ostream& operator<<(std::ostream& os, const ModeFormat& modeFormat) {
  return os << modeFormat.getName();
}


// class ModeTypePack
ModeFormatPack::ModeFormatPack(const std::vector<ModeFormat> modeFormats)
    : modeFormats(modeFormats) {
  for (const auto& modeFormat : modeFormats) {
    taco_uassert(modeFormat.defined()) << "Cannot have undefined mode type";
  }
}

ModeFormatPack::ModeFormatPack(const initializer_list<ModeFormat> modeFormats)
    : modeFormats(modeFormats) {
  for (const auto& modeFormat : modeFormats) {
    taco_uassert(modeFormat.defined()) << "Cannot have undefined mode type";
  }
}

ModeFormatPack::ModeFormatPack(const ModeFormat modeFormat)
    : modeFormats({modeFormat}) {
  taco_uassert(modeFormat.defined()) << "Cannot have undefined mode type";
}

const std::vector<ModeFormat>& ModeFormatPack::getModeFormats() const {
  return modeFormats;
}

bool operator==(const ModeFormatPack& a, const ModeFormatPack& b) {
  const auto aModeTypes = a.getModeFormats();
  const auto bModeTypes = b.getModeFormats();

  if (aModeTypes.size() != bModeTypes.size()) {
    return false;
  }
  for (size_t i = 0; i < aModeTypes.size(); ++i) {
    if (aModeTypes[i] != bModeTypes[i]) {
      return false;
    }
  }
  return true;
}

bool operator!=(const ModeFormatPack& a, const ModeFormatPack& b) {
  return !(a == b);
}

ostream& operator<<(ostream& os, const ModeFormatPack& modeFormatPack) {
  return os << "{" << util::join(modeFormatPack.getModeFormats(), ",") << "}";
}


// Predefined formats
ModeFormat ModeFormat::Dense(std::make_shared<old::DenseModeFormat>());
ModeFormat ModeFormat::Compressed(std::make_shared<CompressedModeFormat>());
ModeFormat ModeFormat::Sparse = ModeFormat::Compressed;
ModeFormat ModeFormat::Singleton(std::make_shared<SingletonModeFormat>());

ModeFormat ModeFormat::dense = ModeFormat::Dense;
ModeFormat ModeFormat::compressed = ModeFormat::Compressed;
ModeFormat ModeFormat::sparse = ModeFormat::Compressed;
ModeFormat ModeFormat::singleton = ModeFormat::Singleton;

const ModeFormat Dense = ModeFormat::Dense;
const ModeFormat Compressed = ModeFormat::Compressed;
const ModeFormat Sparse = ModeFormat::Compressed;
const ModeFormat Singleton = ModeFormat::Singleton;

const ModeFormat dense = ModeFormat::Dense;
const ModeFormat compressed = ModeFormat::Compressed;
const ModeFormat sparse = ModeFormat::Compressed;
const ModeFormat singleton = ModeFormat::Singleton;

const Format CSR({Dense, Sparse}, {0,1});
const Format CSC({Dense, Sparse}, {1,0});
const Format DCSR({Sparse, Sparse}, {0,1});
const Format DCSC({Sparse, Sparse}, {1,0});

const Format COO(int order, bool isUnique, bool isOrdered, bool isAoS, 
                 const std::vector<int>& modeOrdering) {
  taco_uassert(order > 0);
  taco_uassert(modeOrdering.empty() || modeOrdering.size() == (size_t)order);
  taco_iassert(!isAoS);  // TODO: support array-of-structs COO
  
  ModeFormat::Property ordered = isOrdered ? ModeFormat::ORDERED : 
                                 ModeFormat::NOT_ORDERED;
  
  std::vector<ModeFormat> modeTypes;
  modeTypes.push_back(Compressed({ordered, (order == 1 && isUnique) 
                                           ? ModeFormat::UNIQUE 
                                           : ModeFormat::NOT_UNIQUE}));
  if (order > 1) {
    for (int i = 1; i < order - 1; ++i) {
      modeTypes.push_back(Singleton({ordered, ModeFormat::NOT_UNIQUE}));
    }
    modeTypes.push_back(Singleton({ordered, isUnique ? ModeFormat::UNIQUE : 
                                            ModeFormat::NOT_UNIQUE}));
  }
  return modeOrdering.empty() 
         ? Format(modeTypes) 
         : Format(modeTypes, modeOrdering);
}

bool isDense(const Format& format) {
  for (ModeFormat modeFormat : format.getModeFormats()) {
    if (modeFormat != Dense) {
      return false;
    }
  }
  return true;
}

}
