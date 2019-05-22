#ifndef TACO_INTRINSIC_H
#define TACO_INTRINSIC_H

#include <string>
#include <vector>

#include "taco/type.h"
#include "taco/ir/ir.h"

namespace taco {

class IndexExpr;
class Literal;

class Intrinsic {
public:
  virtual ~Intrinsic() {}

  virtual std::string getName() const = 0;
  virtual Datatype inferReturnType(const std::vector<Datatype>&) const = 0;
  virtual ir::Expr lower(const std::vector<ir::Expr>&) const = 0;
  virtual std::vector<size_t> zeroPreservingArgs(const std::vector<IndexExpr>&) const = 0;
};

#define DECLARE_INTRINSIC(NAME) \
  class NAME##Intrinsic : public Intrinsic { \
  public: \
    std::string getName() const; \
    Datatype inferReturnType(const std::vector<Datatype>&) const; \
    ir::Expr lower(const std::vector<ir::Expr>&) const; \
    std::vector<size_t> zeroPreservingArgs(const std::vector<IndexExpr>&) const; \
  };

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
DECLARE_INTRINSIC(Max)
DECLARE_INTRINSIC(Min)
DECLARE_INTRINSIC(Heaviside)

}

#endif

