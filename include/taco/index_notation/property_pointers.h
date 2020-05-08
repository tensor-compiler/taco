#ifndef TACO_PROPERTY_POINTERS_H
#define TACO_PROPERTY_POINTERS_H

#include <vector>
#include <memory>
#include <iostream>
#include <taco/util/intrusive_ptr.h>
#include <taco/util/uncopyable.h>

#include "taco/error.h"
#include "taco/util/comparable.h"

namespace taco {

class Literal;
struct PropertyPtr;

/// A pointer to the property data. This will be wrapped in an auxillary class
/// to allow a user to create a vector of properties. Needed since properties
/// have different methods and data
struct PropertyPtr : public util::Manageable<PropertyPtr>,
                     private util::Uncopyable {
public:
  PropertyPtr();
  virtual ~PropertyPtr();
  virtual std::ostream& print(std::ostream& os) const;
  virtual bool equals(const PropertyPtr* p) const;
};

/// Pointer class for annihilators
struct AnnihilatorPtr : public PropertyPtr {
  AnnihilatorPtr();
  AnnihilatorPtr(Literal);
  AnnihilatorPtr(Literal, std::vector<int>&);

  const Literal& annihilator() const;
  const std::vector<int>& positions() const;

  virtual std::ostream& print(std::ostream& os) const;
  virtual bool equals(const PropertyPtr* p) const;

  struct Content;
  std::shared_ptr<Content> content;
};

/// Pointer class for identities
struct IdentityPtr : public PropertyPtr {
public:
  IdentityPtr();
  IdentityPtr(Literal);
  IdentityPtr(Literal, std::vector<int>&);

  const Literal& identity() const;
  const std::vector<int>& positions() const;

  virtual std::ostream& print(std::ostream& os) const;
  virtual bool equals(const PropertyPtr* p) const;

  struct Content;
  std::shared_ptr<Content> content;
};

/// Pointer class for associativity
struct AssociativePtr : public PropertyPtr {
  AssociativePtr();
  virtual std::ostream& print(std::ostream& os) const;
  virtual bool equals(const PropertyPtr* p) const;
};

/// Pointer class for commutativity
struct CommutativePtr : public PropertyPtr {
  CommutativePtr();
  CommutativePtr(const std::vector<int>&);
  const std::vector<int> ordering_;
  virtual std::ostream& print(std::ostream& os) const;
  virtual bool equals(const PropertyPtr* p) const;
};

template <typename P>
inline bool isa(const PropertyPtr* p) {
  return p != nullptr && dynamic_cast<const P*>(p) != nullptr;
}

template <typename P>
inline const P* to(const PropertyPtr* p) {
  taco_iassert(isa<P>(p)) <<
      "Cannot convert " << typeid(p).name() << " to " << typeid(P).name();;
  return static_cast<const P*>(p);
}

template <typename P>
inline const typename P::Ptr* getPtr(const P& propertyPtr) {
  taco_iassert(isa<typename P::Ptr>(propertyPtr.ptr));
  return static_cast<const typename P::Ptr*>(propertyPtr.ptr);
}


}

#endif //TACO_PROPERTY_POINTERS_H
