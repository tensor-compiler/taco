#include "taco/format.h"

#include <iostream>
#include <climits>
#include <vector>
#include <typeinfo>
#include <initializer_list>

#include "taco/lower/mode_format_dense.h"
#include "taco/lower/mode_format_compressed.h"
#include "taco/lower/mode_format_singleton.h"

#include "taco/error.h"
#include "taco/util/strings.h"

using namespace std;

namespace taco {

// class Format
Format::Format() {
}

Format::Format(const ModeFormat modeFormat) : modeFormatPacks({modeFormat}),
    modeOrdering({0}) {}

Format::Format(const std::initializer_list<ModeFormatPack>& modeFormatPacks)
    : modeFormatPacks(modeFormatPacks) {
  taco_uassert(getOrder() <= INT_MAX) << "Supports only INT_MAX modes";
  
  modeOrdering.resize(getOrder());
  for (int i = 0; i < static_cast<int>(getOrder()); ++i) {
    modeOrdering[i] = i;
  }
}

Format::Format(const std::vector<ModeFormatPack>& modeFormatPacks) :
    modeFormatPacks(modeFormatPacks) {
  taco_uassert(getOrder() <= INT_MAX) << "Supports only INT_MAX modes";
  
  modeOrdering.resize(getOrder());
  for (int i = 0; i < static_cast<int>(getOrder()); ++i) {
    modeOrdering[i] = i;
  }
}

Format::Format(const std::vector<ModeFormatPack>& modeFormatPacks,
               const std::vector<int>& modeOrdering)
    : modeFormatPacks(modeFormatPacks), modeOrdering(modeOrdering) {
  taco_uassert(getOrder() <= INT_MAX) << "Supports only INT_MAX modes";
  taco_uassert((size_t)getOrder() == modeOrdering.size()) <<
      "You must either provide a complete mode ordering or none";
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
      case ZEROLESS:
        if (!isZeroless()) {
          return false;
        }
        break;	
      case PADDED:
        if (!isPadded()) {
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
      case NOT_ZEROLESS:
        if (isZeroless()) {
          return false;
        }
        break;
      case NOT_PADDED:
        if (isPadded()) {
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

bool ModeFormat::isZeroless() const {
  taco_iassert(defined());
  return impl->isZeroless;
}

bool ModeFormat::isPadded() const {
  taco_iassert(defined());
  return impl->isPadded;
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

bool ModeFormat::hasSeqInsertEdge() const {
  taco_iassert(defined());
  return impl->hasSeqInsertEdge;
}

bool ModeFormat::hasInsertCoord() const {
  taco_iassert(defined());
  return impl->hasInsertCoord;
}

bool ModeFormat::isYieldPosPure() const {
  taco_iassert(defined());
  return impl->isYieldPosPure;
}

std::vector<AttrQuery> ModeFormat::getAttrQueries(
    std::vector<IndexVar> parentCoords, 
    std::vector<IndexVar> childCoords) const {
  taco_iassert(defined());
  return impl->attrQueries(parentCoords, childCoords);
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
ModeFormat ModeFormat::Dense(std::make_shared<DenseModeFormat>());
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
  
  std::vector<ModeFormatPack> modeTypes;
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
