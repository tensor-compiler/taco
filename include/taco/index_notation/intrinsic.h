#ifndef TACO_INTRINSIC_H
#define TACO_INTRINSIC_H

#include <string>
#include <vector>

#include "taco/type.h"
#include "taco/ir/ir.h"

namespace taco {

class IndexExpr;

class Intrinsic {
public:
  virtual ~Intrinsic() {}

  /// Returns the name of the instrinsic function.
  virtual std::string getName() const = 0;

  /// Infers the type of the return value based on the types of the arguments.
  virtual Datatype inferReturnType(const std::vector<Datatype>&) const = 0;

  /// Emits IR to compute the value of the intrinsic function.
  virtual ir::Expr lower(const std::vector<ir::Expr>&) const = 0;

  /// Returns a set ZP of zero-preserving argument sets ZP_i.  Each ZP_i 
  /// identifies a set of arguments to the intrinsic function that, if they are 
  /// all zero, forces the result to also be zero.   The zero-preserving 
  /// argument sets must be disjoint (i.e., i != j --> ZP_i \intersect ZP_j == 0).
  virtual std::vector<std::vector<size_t>>
  zeroPreservingArgs(const std::vector<IndexExpr>&) const = 0;
};

#define DECLARE_INTRINSIC(NAME) \
  class NAME##Intrinsic : public Intrinsic { \
  public: \
    std::string getName() const; \
    Datatype inferReturnType(const std::vector<Datatype>&) const; \
    ir::Expr lower(const std::vector<ir::Expr>&) const; \
    std::vector<std::vector<size_t>> zeroPreservingArgs(const std::vector<IndexExpr>&) const; \
  };

DECLARE_INTRINSIC(Mod)
DECLARE_INTRINSIC(Abs)
DECLARE_INTRINSIC(Pow)
DECLARE_INTRINSIC(Square)
DECLARE_INTRINSIC(Cube)
DECLARE_INTRINSIC(Sqrt)
DECLARE_INTRINSIC(Cbrt)
DECLARE_INTRINSIC(Exp)
DECLARE_INTRINSIC(Log)
DECLARE_INTRINSIC(Log10)
DECLARE_INTRINSIC(Sin)
DECLARE_INTRINSIC(Cos)
DECLARE_INTRINSIC(Tan)
DECLARE_INTRINSIC(Asin)
DECLARE_INTRINSIC(Acos)
DECLARE_INTRINSIC(Atan)
DECLARE_INTRINSIC(Atan2)
DECLARE_INTRINSIC(Sinh)
DECLARE_INTRINSIC(Cosh)
DECLARE_INTRINSIC(Tanh)
DECLARE_INTRINSIC(Asinh)
DECLARE_INTRINSIC(Acosh)
DECLARE_INTRINSIC(Atanh)
DECLARE_INTRINSIC(Gt)
DECLARE_INTRINSIC(Lt)
DECLARE_INTRINSIC(Gte)
DECLARE_INTRINSIC(Lte)
DECLARE_INTRINSIC(Eq)
DECLARE_INTRINSIC(Neq)
DECLARE_INTRINSIC(Max)
DECLARE_INTRINSIC(Min)
DECLARE_INTRINSIC(Heaviside)
DECLARE_INTRINSIC(Not)

}

#endif

