#include "taco/index_notation/properties.h"
#include "taco/index_notation/index_notation.h"

namespace taco {

// Property class definitions
Property::Property() : util::IntrusivePtr<const PropertyPtr>(nullptr) {
}

Property::Property(const PropertyPtr* p) : util::IntrusivePtr<const PropertyPtr>(p) {
}

bool Property::equals(const Property &p) const {
  if(!defined() && !p.defined()) {
    return true;
  }

  if(defined() && p.defined()) {
    return ptr->equals(p.ptr);
  }

  return false;
}

std::ostream & Property::print(std::ostream& os) const {
  if(!defined()) {
    os << "Property(undef)";
    return os;
  }
  return ptr->print(os);
}

std::ostream& operator<<(std::ostream& os, const Property& p) {
  return p.print(os);
}

// Annihilator class definitions
template<> bool isa<Annihilator>(const Property& p) {
  return isa<AnnihilatorPtr>(p.ptr);
}

template<> Annihilator to<Annihilator>(const Property& p) {
  taco_iassert(isa<Annihilator>(p));
  return Annihilator(to<AnnihilatorPtr>(p.ptr));
}

Annihilator::Annihilator(Literal annihilator) : Annihilator(new AnnihilatorPtr(annihilator)) {
}

Annihilator::Annihilator(const PropertyPtr* p) : Property(p) {
}

const Literal& Annihilator::annihilator() const {
  taco_iassert(defined());
  return getPtr(*this)->annihilator();
}

// Identity class definitions
template<> bool isa<Identity>(const Property& p) {
  return isa<IdentityPtr>(p.ptr);
}

template<> Identity to<Identity>(const Property& p) {
  taco_iassert(isa<Identity>(p));
  return Identity(to<IdentityPtr>(p.ptr));
}

Identity::Identity(Literal identity) : Identity(new IdentityPtr(identity)) {
}

Identity::Identity(const PropertyPtr* p) : Property(p) {
}

const Literal& Identity::identity() const {
  taco_iassert(defined());
  return getPtr(*this)->identity();
}

// Associative class definitions
template<> bool isa<Associative>(const Property& p) {
  return isa<AssociativePtr>(p.ptr);
}

template<> Associative to<Associative>(const Property& p) {
  taco_iassert(isa<Associative>(p));
  return Associative(to<AssociativePtr>(p.ptr));
}

Associative::Associative() : Associative(new AssociativePtr) {
}

Associative::Associative(const PropertyPtr* p) : Property(p) {
}

// Commutative class definitions
template<> bool isa<Commutative>(const Property& p) {
  return isa<CommutativePtr>(p.ptr);
}

template<> Commutative to<Commutative>(const Property& p) {
  taco_iassert(isa<Commutative>(p));
  return Commutative(to<CommutativePtr>(p.ptr));
}

Commutative::Commutative() : Commutative(new CommutativePtr) {
}

Commutative::Commutative(const std::vector<int>& ordering) : Commutative(new CommutativePtr(ordering)) {
}

Commutative::Commutative(const PropertyPtr* p) : Property(p) {
}

const std::vector<int> & Commutative::ordering() const {
  return getPtr(*this)->ordering_;
}

}