#include "taco/index_notation/intrinsic.h"

#include <complex>
#include <string>

#include "taco/type.h"
#include "taco/ir/ir.h"

namespace taco {

std::string ExpIntrinsic::getName() const {
  return "exp";
}
  
Datatype ExpIntrinsic::inferReturnType(const std::vector<Datatype>& argTypes,
                                       const std::vector<Datatype>& attrTypes) const {
  taco_iassert(argTypes.size() == 1 && attrTypes.empty());
  return argTypes[0];
}

ir::Expr ExpIntrinsic::lower(const std::vector<ir::Expr>& args,
                             const std::vector<ir::Expr>& attrs) const {
  taco_iassert(args.size() == 1 && attrs.empty());

  ir::Expr arg = args[0];
  const bool inZero = (ir::isa<ir::Literal>(arg) && 
                       ir::to<ir::Literal>(arg)->equalsScalar(0.0));

  switch (arg.type().getKind()) {
    case Datatype::Float32:
      return inZero ? ir::Literal::make((float)1.0) : 
             ir::Call::make("expf", args, arg.type());
    case Datatype::Float64:
      return inZero ? ir::Literal::make((double)1.0) : 
             ir::Call::make("exp", args, arg.type());
    case Datatype::Complex64:
      return inZero ? ir::Literal::make(std::complex<float>(1.0)) : 
             ir::Call::make("cexpf", args, arg.type());
    case Datatype::Complex128:
      return inZero ? ir::Literal::make(std::complex<double>(1.0)) : 
             ir::Call::make("cexp", args, arg.type());
    default:
      taco_not_supported_yet;
      break;
  }
  return ir::Expr();
}

bool ExpIntrinsic::isZeroPreserving(const std::vector<Literal>& attrs) const {
  return false;
}

std::string SqrtIntrinsic::getName() const {
  return "sqrt";
}
  
Datatype SqrtIntrinsic::inferReturnType(const std::vector<Datatype>& argTypes,
                                        const std::vector<Datatype>& attrTypes) const {
  taco_iassert(argTypes.size() == 1 && attrTypes.empty());
  return argTypes[0];
}

ir::Expr SqrtIntrinsic::lower(const std::vector<ir::Expr>& args,
                              const std::vector<ir::Expr>& attrs) const {
  taco_iassert(args.size() == 1 && attrs.empty());

  ir::Expr arg = args[0];

  if (ir::isa<ir::Literal>(arg) && 
      ir::to<ir::Literal>(arg)->equalsScalar(0.0)) {
    return ir::Literal::zero(arg.type());
  }

  switch (arg.type().getKind()) {
    case Datatype::Float32:
      return ir::Call::make("sqrtf", args, arg.type());
    case Datatype::Float64:
      return ir::Call::make("sqrt", args, arg.type());
    case Datatype::Complex64:
      return ir::Call::make("csqrtf", args, arg.type());
    case Datatype::Complex128:
      return ir::Call::make("csqrt", args, arg.type());
    default:
      taco_not_supported_yet;
      break;
  }
  return ir::Expr();
}

bool SqrtIntrinsic::isZeroPreserving(const std::vector<Literal>& attrs) const {
  return true;
}

}
