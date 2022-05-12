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

Annihilator::Annihilator(Literal annihilator, std::vector<int> &p) : Annihilator(new AnnihilatorPtr(annihilator, p)) {
}

Annihilator::Annihilator(const PropertyPtr* p) : Property(p) {
}

const Literal& Annihilator::annihilator() const {
  taco_iassert(defined());
  return getPtr(*this)->annihilator();
}

const std::vector<int> & Annihilator::positions() const {
  taco_iassert(defined());
  return getPtr(*this)->positions();
}

IndexExpr Annihilator::annihilates(const std::vector<IndexExpr>& exprs) const {
  taco_iassert(defined());
  Literal a = annihilator();
  std::vector<int> pos = positions();
  if (pos.empty()) {
    for(int i = 0; i < (int)exprs.size(); ++i) {
      pos.push_back(i);
    }
  }

  for(const auto& idx : pos) {
    taco_uassert(idx < (int)exprs.size()) << "Not enough args in expression";
    if(::taco::equals(exprs[idx], a)) {
      return a;
    }
  }

  // We could not simplify.
  return IndexExpr();
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

Identity::Identity(Literal identity, std::vector<int>& positions) : Identity(new IdentityPtr(identity, positions)) {
}

const Literal& Identity::identity() const {
  taco_iassert(defined());
  return getPtr(*this)->identity();
}

const std::vector<int>& Identity::positions() const {
  taco_iassert(defined());
  return getPtr(*this)->positions();
}

IndexExpr Identity::simplify(const std::vector<IndexExpr>& exprs) const {
  // If only one term is not the identity, replace expr with just that term.
  // If all terms are identity, replace with identity.
  Literal identityVal = identity();
  size_t nonIdentityTermsChecked = 0;
  IndexExpr nonIdentityTerm;

  std::vector<int> pos = positions();
  if (pos.empty()) {
    for(int i = 0; i < (int)exprs.size(); ++i) {
      pos.push_back(i);
    }
  }


  for(const auto& idx : pos) {
    if(!::taco::equals(identityVal, exprs[idx])) {
      nonIdentityTerm = exprs[idx];
      ++nonIdentityTermsChecked;
    }
    if(nonIdentityTermsChecked > 1) {
      return IndexExpr();
    }
  }

  size_t identityTermsChecked = pos.size() - nonIdentityTermsChecked;
  if(nonIdentityTermsChecked == 1 && identityTermsChecked == (exprs.size() - 1)) {
    // If we checked all exprs and all are the identity except one return that term
    return nonIdentityTerm;
  }

  if(identityTermsChecked == exprs.size()) {
    // If we checked every expression and
    return identityVal;
  }

  return IndexExpr();
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