#include "taco/index_notation/property_pointers.h"
#include "taco/index_notation/index_notation.h"
#include "taco/util/strings.h"

namespace taco {

struct AnnihilatorPtr::Content {
  Literal annihilator;
  std::vector<int> positions;
};

struct IdentityPtr::Content {
  Literal identity;
  std::vector<int> positions;
};

// Property pointer definitions
PropertyPtr::PropertyPtr() {
}

PropertyPtr::~PropertyPtr() {
}

std::ostream& PropertyPtr::print(std::ostream& os) const {
  os << "Property()";
  return os;
}

bool PropertyPtr::equals(const PropertyPtr* p) const {
  return this == p;
}

// Annihilator pointer definitions
AnnihilatorPtr::AnnihilatorPtr() : PropertyPtr(), content(nullptr) {
}

AnnihilatorPtr::AnnihilatorPtr(Literal annihilator) : PropertyPtr(), content(new Content) {
  content->annihilator = annihilator;
  content->positions = std::vector<int>();
}

AnnihilatorPtr::AnnihilatorPtr(Literal annihilator, std::vector<int>& pos) : PropertyPtr(), content(new Content) {
  content->annihilator = annihilator;
  content->positions = pos;
}

const Literal& AnnihilatorPtr::annihilator() const {
  return content->annihilator;
}

const std::vector<int> & AnnihilatorPtr::positions() const {
  return content->positions;
}

std::ostream& AnnihilatorPtr::print(std::ostream& os) const {
  os << "Annihilator(";
  if (annihilator().defined()) {
    os << annihilator();
  } else {
    os << "undef";
  }
  os << ")";
  return os;
}

bool AnnihilatorPtr::equals(const PropertyPtr* p) const {
  if(!isa<AnnihilatorPtr>(p)) return false;
  const AnnihilatorPtr* a = to<AnnihilatorPtr>(p);
  return ::taco::equals(annihilator(), a->annihilator());
}

// Identity pointer definitions
IdentityPtr::IdentityPtr() : PropertyPtr(), content(nullptr) {
}

IdentityPtr::IdentityPtr(Literal identity) : PropertyPtr(), content(new Content) {
  content->identity = identity;
}

IdentityPtr::IdentityPtr(Literal identity, std::vector<int> &p) : PropertyPtr(), content(new Content) {
  content->identity = identity;
  content->positions = p;
}

const Literal& IdentityPtr::identity() const {
  return content->identity;
}

const std::vector<int> & IdentityPtr::positions() const {
  return content->positions;
}

std::ostream& IdentityPtr::print(std::ostream& os) const {
  os << "Identity(";
  if (identity().defined()) {
    os << identity();
  } else {
    os << "undef";
  }
  os << ")";
  return os;
}

bool IdentityPtr::equals(const PropertyPtr* p) const {
  if(!isa<IdentityPtr>(p)) return false;
  const IdentityPtr* idnty = to<IdentityPtr>(p);
  return ::taco::equals(identity(), idnty->identity());
}

// Associative pointer definitions
AssociativePtr::AssociativePtr() : PropertyPtr() {
}

std::ostream& AssociativePtr::print(std::ostream& os) const {
  os << "Associative()";
  return os;
}

bool AssociativePtr::equals(const PropertyPtr* p) const {
  return isa<AssociativePtr>(p);
}

// CommutativePtr definitions
CommutativePtr::CommutativePtr() : PropertyPtr() {
}

CommutativePtr::CommutativePtr(const std::vector<int>& ordering) : ordering_(ordering) {
}

std::ostream& CommutativePtr::print(std::ostream& os) const {
  os << "Commutative(";
  os << "{" << util::join(ordering_) << "})";
  return os;
}

bool CommutativePtr::equals(const PropertyPtr* p) const {
  if(!isa<CommutativePtr>(p)) return false;
  const CommutativePtr* idnty = to<CommutativePtr>(p);
  return ordering_ == idnty->ordering_;
}

}
