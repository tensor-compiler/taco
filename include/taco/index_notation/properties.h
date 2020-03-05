#ifndef TACO_PROPERTIES_H
#define TACO_PROPERTIES_H

#include <vector>
#include <memory>

#include "taco/error.h"
#include "taco/util/comparable.h"

namespace taco {

class Literal;

class Property {
public:
  virtual ~Property();
  virtual bool defined() const;
  virtual bool equals(const Property&) const;
};

class Annihilator : public Property {
public:
  Annihilator();
  Annihilator(Literal);
  const Literal& getAnnihilator() const;
  virtual bool defined() const;
  virtual bool equals(const Property&) const;

private:
  struct Content;
  std::shared_ptr<Content> content;
};

class Identity : public Property {
public:
  Identity();
  Identity(Literal);
  const Literal& getIdentity() const;
  virtual bool defined() const;
  virtual bool equals(const Property&) const;

private:
  struct Content;
  std::shared_ptr<Content> content;
};


class Associative : public Property {
public:
  Associative();
  static Associative makeUndefined();

  virtual bool defined() const;
  virtual bool equals(const Property&) const;

private:
  bool isDefined;
};

class Commutative : public Property {
public:
  Commutative();
  Commutative(std::vector<int>);
  static Commutative makeUndefined();

  const std::vector<int>& ordering() const;
  virtual bool defined() const;
  virtual bool equals(const Property&) const;

private:
  const std::vector<int> ordering_;
  bool isDefined;
};

/// Returns true if property p is of type P.
template <typename P>
inline bool isa(const Property& p) {
  return dynamic_cast<const P*>(&p) != nullptr;
}

/// Casts the Property p to type P.
template <typename P>
inline const P& to(const Property& p) {
  taco_iassert(isa<P>(p)) << "Cannot convert " << typeid(p).name() << " to " << typeid(P).name();
  return static_cast<const P&>(p);
}

template<typename P>
inline const P findProperty(const std::vector<Property>& properties, P defaultProperty) {
  for (const auto& p: properties) {
    if(isa<P>(p)) return to<P>(p);
  }
  return defaultProperty;
}

}

#endif //TACO_PROPERTIES_H
