#include "taco/index_notation/intrinsic.h"
#include "taco/index_notation/index_notation.h"

#include <complex>
#include <string>
#include <vector>

#include "taco/type.h"
#include "taco/ir/ir.h"

namespace taco {

std::string PowIntrinsic::getName() const {
  return "pow";
}
  
Datatype PowIntrinsic::inferReturnType(const std::vector<Datatype>& argTypes) const {
  taco_iassert(argTypes.size() == 2);
  taco_iassert(argTypes[0] == argTypes[1]);
  return argTypes[0];
}

ir::Expr PowIntrinsic::lower(const std::vector<ir::Expr>& args) const {
  taco_iassert(args.size() == 2);

  ir::Expr base = args[0];
  ir::Expr exponent = args[1];

  if (ir::isa<ir::Literal>(base) && 
      ir::to<ir::Literal>(base)->equalsScalar(0.0)) {
    return ir::Literal::zero(base.type());
  }
  
  const bool exponentZero = (ir::isa<ir::Literal>(exponent) &&
                             ir::to<ir::Literal>(exponent)->equalsScalar(0.0));

  switch (base.type().getKind()) {
    case Datatype::Float32:
      return exponentZero ? ir::Literal::make((float)1.0) : 
             ir::Call::make("powf", args, base.type());
    case Datatype::Float64:
      return exponentZero ? ir::Literal::make((double)1.0) : 
             ir::Call::make("pow", args, base.type());
    case Datatype::Complex64:
      return exponentZero ? ir::Literal::make(std::complex<float>(1.0)) : 
             ir::Call::make("cpowf", args, base.type());
    case Datatype::Complex128:
      return exponentZero ? ir::Literal::make(std::complex<double>(1.0)) : 
             ir::Call::make("cpow", args, base.type());
    default:
      taco_not_supported_yet;
      break;
  }
  return ir::Expr();
}

std::vector<size_t>
PowIntrinsic::zeroPreservingArgs(const std::vector<IndexExpr>& args) const {
  return {0};
}

std::string ExpIntrinsic::getName() const {
  return "exp";
}
  
Datatype ExpIntrinsic::inferReturnType(const std::vector<Datatype>& argTypes) const {
  taco_iassert(argTypes.size() == 1);
  return argTypes[0];
}

ir::Expr ExpIntrinsic::lower(const std::vector<ir::Expr>& args) const {
  taco_iassert(args.size() == 1);

  ir::Expr arg = args[0];
  const bool argZero = (ir::isa<ir::Literal>(arg) && 
                        ir::to<ir::Literal>(arg)->equalsScalar(0.0));

  switch (arg.type().getKind()) {
    case Datatype::Float32:
      return argZero ? ir::Literal::make((float)1.0) : 
             ir::Call::make("expf", args, arg.type());
    case Datatype::Float64:
      return argZero ? ir::Literal::make((double)1.0) : 
             ir::Call::make("exp", args, arg.type());
    case Datatype::Complex64:
      return argZero ? ir::Literal::make(std::complex<float>(1.0)) : 
             ir::Call::make("cexpf", args, arg.type());
    case Datatype::Complex128:
      return argZero ? ir::Literal::make(std::complex<double>(1.0)) : 
             ir::Call::make("cexp", args, arg.type());
    default:
      taco_not_supported_yet;
      break;
  }
  return ir::Expr();
}

std::vector<size_t> 
ExpIntrinsic::zeroPreservingArgs(const std::vector<IndexExpr>& args) const {
  return {};
}

std::string SqrtIntrinsic::getName() const {
  return "sqrt";
}
  
Datatype SqrtIntrinsic::inferReturnType(const std::vector<Datatype>& argTypes) const {
  taco_iassert(argTypes.size() == 1);
  return argTypes[0];
}

ir::Expr SqrtIntrinsic::lower(const std::vector<ir::Expr>& args) const {
  taco_iassert(args.size() == 1);

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

std::vector<size_t>
SqrtIntrinsic::zeroPreservingArgs(const std::vector<IndexExpr>& args) const {
  return {0};
}

std::string MaxIntrinsic::getName() const {
  return "max";
}
  
Datatype MaxIntrinsic::inferReturnType(const std::vector<Datatype>& argTypes) const {
  taco_iassert(argTypes.size() == 2);
  taco_iassert(argTypes[0] == argTypes[1]);
  return argTypes[0];
}

ir::Expr MaxIntrinsic::lower(const std::vector<ir::Expr>& args) const {
  taco_iassert(args.size() == 2);

  ir::Expr a = args[0];
  ir::Expr b = args[1];

  if (ir::isa<ir::Literal>(a) && 
      ir::to<ir::Literal>(a)->equalsScalar(0.0) &&
      ir::isa<ir::Literal>(b) &&
      ir::to<ir::Literal>(b)->equalsScalar(0.0)) {
    return ir::Literal::zero(a.type());
  }

  return ir::Max::make(a, b);
}

std::vector<size_t>
MaxIntrinsic::zeroPreservingArgs(const std::vector<IndexExpr>& args) const {
  return {0, 1};
}

std::string HeavisideIntrinsic::getName() const {
  return "heaviside";
}
  
Datatype HeavisideIntrinsic::inferReturnType(const std::vector<Datatype>& argTypes) const {
  taco_iassert(argTypes.size() == 2);
  taco_iassert(argTypes[0] == argTypes[1]);
  return argTypes[0];
}

ir::Expr HeavisideIntrinsic::lower(const std::vector<ir::Expr>& args) const {
  taco_iassert(args.size() == 2);

  ir::Expr a = args[0];
  ir::Expr b = args[1];

  if (ir::isa<ir::Literal>(a) && 
      ir::to<ir::Literal>(a)->equalsScalar(0.0)) {
    return b;
  }

  ir::Expr zero = ir::Literal::zero(a.type());
  ir::Expr equalsZero = ir::Eq::make(a, zero);
  return ir::Add::make(ir::Mul::make(b, ir::Cast::make(equalsZero, a.type())),
                       ir::Cast::make(ir::Gt::make(a, zero), a.type()));
}

std::vector<size_t>
HeavisideIntrinsic::zeroPreservingArgs(const std::vector<IndexExpr>& args) const {
  taco_iassert(args.size() == 2);

  if (equals(args[1], Literal::zero(args[1].getDataType()))) {
    return {0};
  }

  return {};
}

}
