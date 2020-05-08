#ifndef TACO_PROPERTIES_H
#define TACO_PROPERTIES_H

#include "taco/index_notation/property_pointers.h"
#include "taco/util/intrusive_ptr.h"

namespace taco {

class IndexExpr;

/// A class containing properties about an operation
class Property : public util::IntrusivePtr<const PropertyPtr> {
public:
  Property();
  explicit Property(const PropertyPtr* p);

  bool equals(const Property& p) const;
  std::ostream& print(std::ostream&) const;
};

std::ostream& operator<<(std::ostream&, const Property&);

/// A class wrapping the annihilator property pointer
class Annihilator : public Property {
public:
  explicit Annihilator(Literal);
  Annihilator(Literal, std::vector<int>&);
  explicit Annihilator(const PropertyPtr*);

  const Literal& annihilator() const;
  const std::vector<int>& positions() const;
  IndexExpr annihilates(const std::vector<IndexExpr>&) const;

  typedef AnnihilatorPtr Ptr;
};

/// A class wrapping an identity property pointer
class Identity : public Property {
public:
  explicit Identity(Literal);
  Identity(Literal, std::vector<int>&);
  explicit Identity(const PropertyPtr*);

  const Literal& identity() const;
  const std::vector<int>& positions() const;
  IndexExpr simplify(const std::vector<IndexExpr>&) const;

  typedef IdentityPtr Ptr;
};

/// A class wrapping an associative property pointer
class Associative : public Property {
public:
  Associative();
  explicit Associative(const PropertyPtr*);

  typedef AssociativePtr Ptr;
};

/// A class wrapping a commutative property pointer
class Commutative : public Property {
public:
  Commutative();
  explicit Commutative(const std::vector<int>&);
  explicit Commutative(const PropertyPtr*);

  const std::vector<int>& ordering() const;

  typedef CommutativePtr Ptr;
};

/// Returns true if property p is of type P.
template <typename P> bool isa(const Property& p);

/// Casts the Property p to type P.
template <typename P> P to(const Property& p);

/// Finds and returns the property of type P if it exists in the vector. If
/// the property does not exist, returns an undefined instance of the property
/// requested.
/// The vector of properties should not contain duplicates so this is sufficient.
template<typename P>
inline const P findProperty(const std::vector<Property> &properties) {
  for (const auto &p: properties) {
    if (isa<P>(p)) return to<P>(p);
  }
  return P(nullptr);
}

}

#endif //TACO_PROPERTIES_H
