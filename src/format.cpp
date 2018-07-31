#include "taco/format.h"

#include <iostream>
#include <climits>
#include <vector>
#include <initializer_list>

#include "taco/lower/mode_format_dense.h"
#include "taco/lower/mode_format_compressed.h"

#include "taco/error.h"
#include "taco/util/strings.h"

namespace taco {


// class Format
Format::Format() {
}

Format::Format(const ModeFormat modeType) : modeTypePacks({modeType}),
    modeOrdering({0}) {}

Format::Format(const std::initializer_list<ModeFormatPack>& modeTypePacks) :
    modeTypePacks(modeTypePacks) {
  taco_uassert(getOrder() <= INT_MAX) << "Supports only INT_MAX modes";
  
  modeOrdering.resize(getOrder());
  for (int i = 0; i < static_cast<int>(getOrder()); ++i) {
    modeOrdering[i] = i;
  }
}

Format::Format(const std::vector<ModeFormatPack>& modeTypePacks) : 
    modeTypePacks(modeTypePacks) {
  taco_uassert(getOrder() <= INT_MAX) << "Supports only INT_MAX modes";
  
  modeOrdering.resize(getOrder());
  for (int i = 0; i < static_cast<int>(getOrder()); ++i) {
    modeOrdering[i] = i;
  }
}

Format::Format(const std::vector<ModeFormatPack>& modeTypePacks,
               const std::vector<size_t>& modeOrdering) : 
    modeTypePacks(modeTypePacks), modeOrdering(modeOrdering) {
  taco_uassert(getOrder() <= INT_MAX) << "Supports only INT_MAX modes";
  taco_uassert(getOrder() == modeOrdering.size()) <<
      "You must either provide a complete mode ordering or none";
}

size_t Format::getOrder() const {
  return getModeTypes().size();
}

const std::vector<ModeFormat> Format::getModeTypes() const {
  std::vector<ModeFormat> modeTypes;
  for (const auto modeTypePack : getModeTypePacks()) {
    modeTypes.insert(modeTypes.end(), modeTypePack.getModeTypes().begin(),
                     modeTypePack.getModeTypes().end());
  }
  return modeTypes;
}

const std::vector<ModeFormatPack>& Format::getModeTypePacks() const {
  return this->modeTypePacks;
}

const std::vector<size_t>& Format::getModeOrdering() const {
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
  if (getModeTypes()[level] == Sparse) {
    return levelArrayTypes[level][1];
  }
  return levelArrayTypes[level][0];
}

void Format::setLevelArrayTypes(std::vector<std::vector<Datatype>> levelArrayTypes) {
  this->levelArrayTypes = levelArrayTypes;
}


bool operator==(const Format& a, const Format& b){
  const auto aModeTypePacks = a.getModeTypePacks();
  const auto bModeTypePacks = b.getModeTypePacks();
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
  return os << "(" << util::join(format.getModeTypePacks(), ",") << "; "
            << util::join(format.getModeOrdering(), ",") << ")";
}


// class ModeType
ModeFormat::ModeFormat() {
}

ModeFormat::ModeFormat(const std::shared_ptr<ModeFormatImpl> impl) : impl(impl) {
}

ModeFormat ModeFormat::operator()(const std::vector<Property>& properties) {
  return defined() ? impl->copy(properties) : ModeFormat();
}

std::string ModeFormat::getName() const {
  return defined() ? impl->name : "undefined";
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

bool ModeFormat::defined() const {
  return impl != nullptr;
}

bool operator==(const ModeFormat& a, const ModeFormat& b) {
  return (a.defined() && b.defined() &&
          a.getName() == b.getName() &&
          a.isFull() == b.isFull() &&
          a.isOrdered() == b.isOrdered() &&
          a.isUnique() == b.isUnique() &&
          a.isBranchless() == b.isBranchless() &&
          a.isCompact() == b.isCompact());
}

bool operator!=(const ModeFormat& a, const ModeFormat& b) {
  return !(a == b);
}

std::ostream& operator<<(std::ostream& os, const ModeFormat& modeType) {
  return os << modeType.getName();
}


// class ModeTypePack
ModeFormatPack::ModeFormatPack(const std::vector<ModeFormat> modeTypes)
    : modeTypes(modeTypes) {
  for (const auto& modeType : modeTypes) {
    taco_uassert(modeType.defined()) << "Cannot have undefined mode type";
  }
}

ModeFormatPack::ModeFormatPack(const std::initializer_list<ModeFormat> modeTypes) :
    modeTypes(modeTypes) {
  for (const auto& modeType : modeTypes) {
    taco_uassert(modeType.defined()) << "Cannot have undefined mode type";
  }
}

ModeFormatPack::ModeFormatPack(const ModeFormat modeType) : modeTypes({modeType}) {
  taco_uassert(modeType.defined()) << "Cannot have undefined mode type";
}

const std::vector<ModeFormat>& ModeFormatPack::getModeTypes() const {
  return modeTypes;
}

bool operator==(const ModeFormatPack& a, const ModeFormatPack& b) {
  const auto aModeTypes = a.getModeTypes();
  const auto bModeTypes = b.getModeTypes();

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

std::ostream& operator<<(std::ostream& os, const ModeFormatPack& modeTypePack) {
  return os << "{" << util::join(modeTypePack.getModeTypes(), ",") << "}";
}


// Predefined formats
ModeFormat ModeFormat::Dense(std::make_shared<DenseModeFormat>());
ModeFormat ModeFormat::Compressed(std::make_shared<CompressedModeFormat>());
ModeFormat ModeFormat::Sparse = ModeFormat::Compressed;

ModeFormat ModeFormat::dense = ModeFormat::Dense;
ModeFormat ModeFormat::compressed = ModeFormat::Compressed;
ModeFormat ModeFormat::sparse = ModeFormat::Compressed;

const ModeFormat Dense = ModeFormat::Dense;
const ModeFormat Compressed = ModeFormat::Compressed;
const ModeFormat Sparse = ModeFormat::Compressed;

const ModeFormat dense = ModeFormat::Dense;
const ModeFormat compressed = ModeFormat::Compressed;
const ModeFormat sparse = ModeFormat::Compressed;

const Format CSR({Dense, Sparse}, {0,1});
const Format CSC({Dense, Sparse}, {1,0});
const Format DCSR({Sparse, Sparse}, {0,1});
const Format DCSC({Sparse, Sparse}, {1,0});

bool isDense(const Format& format) {
  for (ModeFormat modeType : format.getModeTypes()) {
    if (modeType != Dense) {
      return false;
    }
  }
  return true;
}

}
