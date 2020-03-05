#include "taco/index_notation/properties.h"
#include "taco/index_notation/index_notation.h"

namespace taco {

struct Annihilator::Content {
  Literal annihilator;
};

struct Identity::Content {
  Literal identity;
};

// Property class definitions
Property::~Property() {}

bool Property::defined() const {
  return false;
}

bool Property::equals(const Property& p) const {
  return defined() == p.defined();
}

// Annihilator class definitions
Annihilator::Annihilator() {}

Annihilator::Annihilator(Literal annihilator) : content(new Content) {
  content->annihilator = annihilator;
}

const Literal& Annihilator::getAnnihilator() const {
  taco_iassert(defined());
  return content->annihilator;
}

bool Annihilator::defined() const {
  return content.get() != nullptr;
}

bool Annihilator::equals(const Property& p) const {
  if(!isa<Annihilator>(p)) return false;

  Annihilator a = to<Annihilator>(p);
  if (!defined() && !a.defined()) return true;

  if(defined() && a.defined()) {
    return ::taco::equals(getAnnihilator(), a.getAnnihilator());
  }
  return false;
}

// Identity class definitions
Identity::Identity() {}

Identity::Identity(Literal identity) : content(new Content) {
  content->identity = identity;
}

const Literal& Identity::getIdentity() const {
  taco_iassert(defined());
  return content->identity;
}

bool Identity::defined() const {
  return content.get() != nullptr;
}

bool Identity::equals(const Property& p) const {
  if(!isa<Identity>(p)) return false;

  Identity i = to<Identity>(p);
  if (!defined() && !i.defined()) return true;

  if(defined() && i.defined()) {
    return ::taco::equals(getIdentity(), i.getIdentity());
  }
  return false;
}

// Associative class definitions
Associative::Associative() : isDefined(true) {}

Associative Associative::makeUndefined() {
  Associative a = Associative();
  a.isDefined = false;
  return a;
}

bool Associative::defined() const {
  return isDefined;
}

bool Associative::equals(const Property& p) const {
  if(!isa<Associative>(p)) return false;
  Associative a = to<Associative>(p);
  return defined() == a.defined();
}

// Commutative class definitions
Commutative::Commutative() : isDefined(true) {}

Commutative::Commutative(std::vector<int> ordering) : ordering_(ordering), isDefined(true) {
}

Commutative Commutative::makeUndefined() {
  Commutative com;
  com.isDefined = false;
  return com;
}

const std::vector<int> & Commutative::ordering() const {
  return ordering_;
}

bool Commutative::defined() const {
  return isDefined;
}

bool Commutative::equals(const Property& p) const {
  if(!isa<Commutative>(p)) return false;

  Commutative c = to<Commutative>(p);
  if (!defined() && !c.defined()) return true;

  if(defined() && c.defined()) {
    return ordering() == c.ordering();
  }

  return false;
}

}