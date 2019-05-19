#ifndef TACO_INTRINSIC_H
#define TACO_INTRINSIC_H

#include <vector>
#include <string>

#include "taco/type.h"
#include "taco/ir/ir.h"

namespace taco {

class IndexExpr;
class Literal;

class Intrinsic {
public:
  virtual ~Intrinsic() {}

  virtual std::string getName() const = 0;
  virtual Datatype inferReturnType(const std::vector<Datatype>&,
                                   const std::vector<Datatype>&) const = 0;
  virtual ir::Expr lower(const std::vector<ir::Expr>&,
                         const std::vector<ir::Expr>&) const = 0;
  virtual bool isZeroPreserving(const std::vector<Literal>&) const = 0;
};

class SqrtIntrinsic : public Intrinsic {
public:
  std::string getName() const;
  Datatype inferReturnType(const std::vector<Datatype>&,
                           const std::vector<Datatype>&) const;
  ir::Expr lower(const std::vector<ir::Expr>&,
                 const std::vector<ir::Expr>&) const;
  bool isZeroPreserving(const std::vector<Literal>&) const;
};

class ExpIntrinsic : public Intrinsic {
public:
  std::string getName() const;
  Datatype inferReturnType(const std::vector<Datatype>&,
                           const std::vector<Datatype>&) const;
  ir::Expr lower(const std::vector<ir::Expr>&,
                 const std::vector<ir::Expr>&) const;
  bool isZeroPreserving(const std::vector<Literal>&) const;
};

}

#endif

