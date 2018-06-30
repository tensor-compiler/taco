#include "taco/format.h"

#include <iostream>
#include <climits>

#include "taco/error.h"
#include "taco/util/strings.h"

namespace taco {

// class ModeTypePack
ModeTypePack::ModeTypePack(const std::vector<ModeType> modeTypes)
    : modeTypes(modeTypes) {
  for (const auto& modeType : modeTypes) {
    taco_uassert(modeType.defined()) << "Cannot have undefined mode type";
  }
}

ModeTypePack::ModeTypePack(const std::initializer_list<ModeType> modeTypes) : 
    modeTypes(modeTypes) {
  for (const auto& modeType : modeTypes) {
    taco_uassert(modeType.defined()) << "Cannot have undefined mode type";
  }
}

ModeTypePack::ModeTypePack(const ModeType modeType) : modeTypes({modeType}) {
  taco_uassert(modeType.defined()) << "Cannot have undefined mode type";
}

const std::vector<ModeType>& ModeTypePack::getModeTypes() const {
  return modeTypes;
}

// class Format
Format::Format() {
}

Format::Format(const ModeType modeType) : modeTypePacks({modeType}),
    modeOrdering({0}) {}

Format::Format(const std::vector<ModeTypePack>& modeTypePacks) : 
    modeTypePacks(modeTypePacks) {
  taco_uassert(getOrder() <= INT_MAX) << "Supports only INT_MAX modes";
  
  modeOrdering.resize(getOrder());
  for (int i = 0; i < static_cast<int>(getOrder()); ++i) {
    modeOrdering[i] = i;
  }
}

Format::Format(const std::vector<ModeTypePack>& modeTypePacks,
               const std::vector<size_t>& modeOrdering) : 
    modeTypePacks(modeTypePacks), modeOrdering(modeOrdering) {
  taco_uassert(getOrder() <= INT_MAX) << "Supports only INT_MAX modes";
  taco_uassert(getOrder() == modeOrdering.size()) <<
      "You must either provide a complete mode ordering or none";
}

size_t Format::getOrder() const {
  return getModeTypes().size();
}

const std::vector<ModeType> Format::getModeTypes() const {
  std::vector<ModeType> modeTypes;
  for (const auto modeTypePack : getModeTypePacks()) {
    modeTypes.insert(modeTypes.end(), modeTypePack.getModeTypes().begin(),
                     modeTypePack.getModeTypes().end());
  }
  return modeTypes;
}

const std::vector<ModeTypePack>& Format::getModeTypePacks() const {
  return this->modeTypePacks;
}

const std::vector<size_t>& Format::getModeOrdering() const {
  return this->modeOrdering;
}

const std::vector<std::vector<Datatype>>& Format::getLevelArrayTypes() const {
  return this->levelArrayTypes;
}

Datatype Format::getCoordinateTypePos(int level) const {
  return levelArrayTypes[level][0];
}

Datatype Format::getCoordinateTypeIdx(int level) const {
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

bool operator==(const ModeTypePack& a, const ModeTypePack& b) {
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

bool operator!=(const ModeTypePack& a, const ModeTypePack& b) {
  return !(a == b);
}

std::ostream &operator<<(std::ostream& os, const Format& format) {
  return os << "(" << util::join(format.getModeTypePacks(), ",") << "; "
            << util::join(format.getModeOrdering(), ",") << ")";
}

std::ostream& operator<<(std::ostream& os, const ModeTypePack& modeTypePack) {
  return os << "{" << util::join(modeTypePack.getModeTypes(), ",") << "}";
}

// Predefined formats
ModeType ModeType::Dense(std::make_shared<DenseModeType>());
ModeType ModeType::Compressed(std::make_shared<CompressedModeType>());
ModeType ModeType::Sparse = ModeType::Compressed;

ModeType ModeType::dense = ModeType::Dense;
ModeType ModeType::compressed = ModeType::Compressed;
ModeType ModeType::sparse = ModeType::Compressed;

const ModeType Dense = ModeType::Dense;
const ModeType Compressed = ModeType::Compressed;
const ModeType Sparse = ModeType::Compressed;

const ModeType dense = ModeType::Dense;
const ModeType compressed = ModeType::Compressed;
const ModeType sparse = ModeType::Compressed;

const Format CSR({Dense, Sparse}, {0,1});
const Format CSC({Dense, Sparse}, {1,0});
const Format DCSR({Sparse, Sparse}, {0,1});
const Format DCSC({Sparse, Sparse}, {1,0});

bool isDense(const Format& format) {
  for (ModeType modeType : format.getModeTypes()) {
    if (modeType != Dense) {
      return false;
    }
  }
  return true;
}

}
