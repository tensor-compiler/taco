#include "taco/index_notation/intrinsic.h"
#include "taco/index_notation/index_notation.h"

#include <complex>
#include <string>
#include <vector>

#include "taco/type.h"
#include "taco/ir/ir.h"

namespace taco {

// class ModIntrinsic

std::string ModIntrinsic::getName() const {
  return "mod";
}
  
Datatype ModIntrinsic::inferReturnType(const std::vector<Datatype>& argTypes) const {
  taco_iassert(argTypes.size() == 2);
  taco_iassert(argTypes[0] == argTypes[1]);
  return argTypes[0];
}

ir::Expr ModIntrinsic::lower(const std::vector<ir::Expr>& args) const {
  taco_iassert(args.size() == 2);

  ir::Expr a = args[0];
  ir::Expr b = args[1];

  if (ir::isa<ir::Literal>(a) && 
      ir::to<ir::Literal>(a)->equalsScalar(0.0)) {
    return a;
  }

  switch (a.type().getKind()) {
    case Datatype::UInt8:
    case Datatype::UInt16:
    case Datatype::UInt32:
    case Datatype::UInt64:
    case Datatype::Int8: 
    case Datatype::Int16:
    case Datatype::Int32:
    case Datatype::Int64:
      return ir::Rem::make(a, b);
    case Datatype::Float32:
      return ir::Call::make("fmodf", args, a.type());
    case Datatype::Float64:
      return ir::Call::make("fmod", args, a.type());
    default:
      taco_not_supported_yet;
      break;
  }
  return ir::Expr();
}

std::vector<std::vector<size_t>>
ModIntrinsic::zeroPreservingArgs(const std::vector<IndexExpr>& args) const {
  return {{0}, {1}};
}


// class AbsIntrinsic

std::string AbsIntrinsic::getName() const {
  return "abs";
}
  
Datatype AbsIntrinsic::inferReturnType(const std::vector<Datatype>& argTypes) const {
  taco_iassert(argTypes.size() == 1);
  return argTypes[0];
}

ir::Expr AbsIntrinsic::lower(const std::vector<ir::Expr>& args) const {
  taco_iassert(args.size() == 1);

  ir::Expr arg = args[0];
  
  if (ir::isa<ir::Literal>(arg) && 
      ir::to<ir::Literal>(arg)->equalsScalar(0.0)) {
    return arg;
  }
  
  switch (arg.type().getKind()) {
    case Datatype::UInt8:
    case Datatype::UInt16:
    case Datatype::UInt32:
    case Datatype::UInt64:
      return arg;
    case Datatype::Int8: 
    case Datatype::Int16:
    case Datatype::Int32:
      return ir::Call::make("abs", args, arg.type());
    case Datatype::Int64:
      return ir::Call::make("labs", args, arg.type());
    case Datatype::Float32:
      return ir::Call::make("fabsf", args, arg.type());
    case Datatype::Float64:
      return ir::Call::make("fabs", args, arg.type());
    case Datatype::Complex64:
      return ir::Call::make("cabsf", args, arg.type());
    case Datatype::Complex128:
      return ir::Call::make("cabs", args, arg.type());
    default:
      taco_not_supported_yet;
      break;
  }
  return ir::Expr();
}

std::vector<std::vector<size_t>> 
AbsIntrinsic::zeroPreservingArgs(const std::vector<IndexExpr>& args) const {
  return {{0}};
}


// class PowIntrinsic

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

  if ((ir::isa<ir::Literal>(base) && 
       ir::to<ir::Literal>(base)->equalsScalar(0.0)) ||
      (ir::isa<ir::Literal>(exponent) && 
       ir::to<ir::Literal>(exponent)->equalsScalar(1.0))) {
    return base;
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

std::vector<std::vector<size_t>>
PowIntrinsic::zeroPreservingArgs(const std::vector<IndexExpr>& args) const {
  return {{0}};
}


// class SquareIntrinsic

std::string SquareIntrinsic::getName() const {
  return "square";
}
  
Datatype SquareIntrinsic::inferReturnType(const std::vector<Datatype>& argTypes) const {
  taco_iassert(argTypes.size() == 1);
  return argTypes[0];
}

ir::Expr SquareIntrinsic::lower(const std::vector<ir::Expr>& args) const {
  taco_iassert(args.size() == 1);

  ir::Expr arg = args[0];
  
  if (ir::isa<ir::Literal>(arg) && 
      ir::to<ir::Literal>(arg)->equalsScalar(0.0)) {
    return arg;
  }
  
  return ir::Mul::make(arg, arg);
}

std::vector<std::vector<size_t>> 
SquareIntrinsic::zeroPreservingArgs(const std::vector<IndexExpr>& args) const {
  return {{0}};
}


// class CubeIntrinsic

std::string CubeIntrinsic::getName() const {
  return "cube";
}
  
Datatype CubeIntrinsic::inferReturnType(const std::vector<Datatype>& argTypes) const {
  taco_iassert(argTypes.size() == 1);
  return argTypes[0];
}

ir::Expr CubeIntrinsic::lower(const std::vector<ir::Expr>& args) const {
  taco_iassert(args.size() == 1);

  ir::Expr arg = args[0];
  
  if (ir::isa<ir::Literal>(arg) && 
      ir::to<ir::Literal>(arg)->equalsScalar(0.0)) {
    return arg;
  }
  
  return ir::Mul::make(ir::Mul::make(arg, arg), arg);
}

std::vector<std::vector<size_t>> 
CubeIntrinsic::zeroPreservingArgs(const std::vector<IndexExpr>& args) const {
  return {{0}};
}


// class SqrtIntrinsic

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
      (ir::to<ir::Literal>(arg)->equalsScalar(0.0) ||
       ir::to<ir::Literal>(arg)->equalsScalar(1.0))) {
    return arg;
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

std::vector<std::vector<size_t>>
SqrtIntrinsic::zeroPreservingArgs(const std::vector<IndexExpr>& args) const {
  return {{0}};
}


// class CbrtIntrinsic

std::string CbrtIntrinsic::getName() const {
  return "cbrt";
}
  
Datatype CbrtIntrinsic::inferReturnType(const std::vector<Datatype>& argTypes) const {
  taco_iassert(argTypes.size() == 1);
  return argTypes[0];
}

ir::Expr CbrtIntrinsic::lower(const std::vector<ir::Expr>& args) const {
  taco_iassert(args.size() == 1);

  ir::Expr arg = args[0];

  if (ir::isa<ir::Literal>(arg) && 
      (ir::to<ir::Literal>(arg)->equalsScalar(0.0) ||
       ir::to<ir::Literal>(arg)->equalsScalar(1.0) ||
       ir::to<ir::Literal>(arg)->equalsScalar(-1.0))) {
    return arg;
  }

  switch (arg.type().getKind()) {
    case Datatype::Float32:
      return ir::Call::make("cbrtf", args, arg.type());
    case Datatype::Float64:
      return ir::Call::make("cbrt", args, arg.type());
    default:
      taco_not_supported_yet;
      break;
  }
  return ir::Expr();
}

std::vector<std::vector<size_t>>
CbrtIntrinsic::zeroPreservingArgs(const std::vector<IndexExpr>& args) const {
  return {{0}};
}


// class ExpIntrinsic

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

std::vector<std::vector<size_t>> 
ExpIntrinsic::zeroPreservingArgs(const std::vector<IndexExpr>& args) const {
  return {};
}


// class LogIntrinsic

std::string LogIntrinsic::getName() const {
  return "log";
}
  
Datatype LogIntrinsic::inferReturnType(const std::vector<Datatype>& argTypes) const {
  taco_iassert(argTypes.size() == 1);
  return argTypes[0];
}

ir::Expr LogIntrinsic::lower(const std::vector<ir::Expr>& args) const {
  taco_iassert(args.size() == 1);

  ir::Expr arg = args[0];
  
  if (ir::isa<ir::Literal>(arg) && 
      ir::to<ir::Literal>(arg)->equalsScalar(1.0)) {
    return ir::Literal::zero(arg.type());
  }
  
  switch (arg.type().getKind()) {
    case Datatype::Float32:
      return ir::Call::make("logf", args, arg.type());
    case Datatype::Float64:
      return ir::Call::make("log", args, arg.type());
    case Datatype::Complex64:
      return ir::Call::make("clogf", args, arg.type());
    case Datatype::Complex128:
      return ir::Call::make("clog", args, arg.type());
    default:
      taco_not_supported_yet;
      break;
  }
  return ir::Expr();
}

std::vector<std::vector<size_t>> 
LogIntrinsic::zeroPreservingArgs(const std::vector<IndexExpr>& args) const {
  return {};
}


// class Log10Intrinsic

std::string Log10Intrinsic::getName() const {
  return "log10";
}
  
Datatype Log10Intrinsic::inferReturnType(const std::vector<Datatype>& argTypes) const {
  taco_iassert(argTypes.size() == 1);
  return argTypes[0];
}

ir::Expr Log10Intrinsic::lower(const std::vector<ir::Expr>& args) const {
  taco_iassert(args.size() == 1);

  ir::Expr arg = args[0];
  
  if (ir::isa<ir::Literal>(arg) && 
      ir::to<ir::Literal>(arg)->equalsScalar(1.0)) {
    return ir::Literal::zero(arg.type());
  }
  
  switch (arg.type().getKind()) {
    case Datatype::Float32:
    case Datatype::Float64:
      return ir::Call::make("log10", args, arg.type());
    default:
      taco_not_supported_yet;
      break;
  }
  return ir::Expr();
}

std::vector<std::vector<size_t>> 
Log10Intrinsic::zeroPreservingArgs(const std::vector<IndexExpr>& args) const {
  return {};
}


// class SinIntrinsic

std::string SinIntrinsic::getName() const {
  return "sin";
}
  
Datatype SinIntrinsic::inferReturnType(const std::vector<Datatype>& argTypes) const {
  taco_iassert(argTypes.size() == 1);
  return argTypes[0];
}

ir::Expr SinIntrinsic::lower(const std::vector<ir::Expr>& args) const {
  taco_iassert(args.size() == 1);

  ir::Expr arg = args[0];

  if (ir::isa<ir::Literal>(arg) && 
      (ir::to<ir::Literal>(arg)->equalsScalar(0.0))) {
    return arg;
  }

  switch (arg.type().getKind()) {
    case Datatype::Float32:
      return ir::Call::make("sinf", args, arg.type());
    case Datatype::Float64:
      return ir::Call::make("sin", args, arg.type());
    case Datatype::Complex64:
      return ir::Call::make("csinf", args, arg.type());
    case Datatype::Complex128:
      return ir::Call::make("csin", args, arg.type());
    default:
      taco_not_supported_yet;
      break;
  }
  return ir::Expr();
}

std::vector<std::vector<size_t>>
SinIntrinsic::zeroPreservingArgs(const std::vector<IndexExpr>& args) const {
  return {{0}};
}


// class CosIntrinsic

std::string CosIntrinsic::getName() const {
  return "cos";
}
  
Datatype CosIntrinsic::inferReturnType(const std::vector<Datatype>& argTypes) const {
  taco_iassert(argTypes.size() == 1);
  return argTypes[0];
}

ir::Expr CosIntrinsic::lower(const std::vector<ir::Expr>& args) const {
  taco_iassert(args.size() == 1);

  ir::Expr arg = args[0];
  const bool argZero = (ir::isa<ir::Literal>(arg) && 
                        ir::to<ir::Literal>(arg)->equalsScalar(0.0));

  switch (arg.type().getKind()) {
    case Datatype::Float32:
      return argZero ? ir::Literal::make((float)1.0) : 
             ir::Call::make("cosf", args, arg.type());
    case Datatype::Float64:
      return argZero ? ir::Literal::make((double)1.0) : 
             ir::Call::make("cos", args, arg.type());
    case Datatype::Complex64:
      return argZero ? ir::Literal::make(std::complex<float>(1.0)) : 
             ir::Call::make("ccosf", args, arg.type());
    case Datatype::Complex128:
      return argZero ? ir::Literal::make(std::complex<double>(1.0)) : 
             ir::Call::make("ccos", args, arg.type());
    default:
      taco_not_supported_yet;
      break;
  }
  return ir::Expr();
}

std::vector<std::vector<size_t>> 
CosIntrinsic::zeroPreservingArgs(const std::vector<IndexExpr>& args) const {
  return {};
}


// class TanIntrinsic

std::string TanIntrinsic::getName() const {
  return "tan";
}
  
Datatype TanIntrinsic::inferReturnType(const std::vector<Datatype>& argTypes) const {
  taco_iassert(argTypes.size() == 1);
  return argTypes[0];
}

ir::Expr TanIntrinsic::lower(const std::vector<ir::Expr>& args) const {
  taco_iassert(args.size() == 1);

  ir::Expr arg = args[0];

  if (ir::isa<ir::Literal>(arg) && 
      (ir::to<ir::Literal>(arg)->equalsScalar(0.0))) {
    return arg;
  }

  switch (arg.type().getKind()) {
    case Datatype::Float32:
      return ir::Call::make("tanf", args, arg.type());
    case Datatype::Float64:
      return ir::Call::make("tan", args, arg.type());
    case Datatype::Complex64:
      return ir::Call::make("ctanf", args, arg.type());
    case Datatype::Complex128:
      return ir::Call::make("ctan", args, arg.type());
    default:
      taco_not_supported_yet;
      break;
  }
  return ir::Expr();
}

std::vector<std::vector<size_t>>
TanIntrinsic::zeroPreservingArgs(const std::vector<IndexExpr>& args) const {
  return {{0}};
}


// class AsinIntrinsic

std::string AsinIntrinsic::getName() const {
  return "asin";
}
  
Datatype AsinIntrinsic::inferReturnType(const std::vector<Datatype>& argTypes) const {
  taco_iassert(argTypes.size() == 1);
  return argTypes[0];
}

ir::Expr AsinIntrinsic::lower(const std::vector<ir::Expr>& args) const {
  taco_iassert(args.size() == 1);

  ir::Expr arg = args[0];

  if (ir::isa<ir::Literal>(arg) && 
      (ir::to<ir::Literal>(arg)->equalsScalar(0.0))) {
    return arg;
  }

  switch (arg.type().getKind()) {
    case Datatype::Float32:
      return ir::Call::make("asinf", args, arg.type());
    case Datatype::Float64:
      return ir::Call::make("asin", args, arg.type());
    case Datatype::Complex64:
      return ir::Call::make("casinf", args, arg.type());
    case Datatype::Complex128:
      return ir::Call::make("casin", args, arg.type());
    default:
      taco_not_supported_yet;
      break;
  }
  return ir::Expr();
}

std::vector<std::vector<size_t>>
AsinIntrinsic::zeroPreservingArgs(const std::vector<IndexExpr>& args) const {
  return {{0}};
}


// class AcosIntrinsic

std::string AcosIntrinsic::getName() const {
  return "acos";
}
  
Datatype AcosIntrinsic::inferReturnType(const std::vector<Datatype>& argTypes) const {
  taco_iassert(argTypes.size() == 1);
  return argTypes[0];
}

ir::Expr AcosIntrinsic::lower(const std::vector<ir::Expr>& args) const {
  taco_iassert(args.size() == 1);

  ir::Expr arg = args[0];

  switch (arg.type().getKind()) {
    case Datatype::Float32:
      return ir::Call::make("acosf", args, arg.type());
    case Datatype::Float64:
      return ir::Call::make("acos", args, arg.type());
    case Datatype::Complex64:
      return ir::Call::make("cacosf", args, arg.type());
    case Datatype::Complex128:
      return ir::Call::make("cacos", args, arg.type());
    default:
      taco_not_supported_yet;
      break;
  }
  return ir::Expr();
}

std::vector<std::vector<size_t>> 
AcosIntrinsic::zeroPreservingArgs(const std::vector<IndexExpr>& args) const {
  return {};
}


// class AtanIntrinsic

std::string AtanIntrinsic::getName() const {
  return "atan";
}
  
Datatype AtanIntrinsic::inferReturnType(const std::vector<Datatype>& argTypes) const {
  taco_iassert(argTypes.size() == 1);
  return argTypes[0];
}

ir::Expr AtanIntrinsic::lower(const std::vector<ir::Expr>& args) const {
  taco_iassert(args.size() == 1);

  ir::Expr arg = args[0];

  if (ir::isa<ir::Literal>(arg) && 
      (ir::to<ir::Literal>(arg)->equalsScalar(0.0))) {
    return arg;
  }

  switch (arg.type().getKind()) {
    case Datatype::Float32:
      return ir::Call::make("atanf", args, arg.type());
    case Datatype::Float64:
      return ir::Call::make("atan", args, arg.type());
    case Datatype::Complex64:
      return ir::Call::make("catanf", args, arg.type());
    case Datatype::Complex128:
      return ir::Call::make("catan", args, arg.type());
    default:
      taco_not_supported_yet;
      break;
  }
  return ir::Expr();
}

std::vector<std::vector<size_t>>
AtanIntrinsic::zeroPreservingArgs(const std::vector<IndexExpr>& args) const {
  return {{0}};
}


// class Atan2Intrinsic

std::string Atan2Intrinsic::getName() const {
  return "atan2";
}
  
Datatype Atan2Intrinsic::inferReturnType(const std::vector<Datatype>& argTypes) const {
  taco_iassert(argTypes.size() == 2);
  taco_iassert(argTypes[0] == argTypes[1]);
  return argTypes[0];
}

ir::Expr Atan2Intrinsic::lower(const std::vector<ir::Expr>& args) const {
  taco_iassert(args.size() == 2);

  ir::Expr a = args[0];
  ir::Expr b = args[1];

  if (ir::isa<ir::Literal>(a) && 
      ir::to<ir::Literal>(a)->equalsScalar(0.0) &&
      ir::isa<ir::Literal>(b) &&
      ir::to<ir::Literal>(b)->equalsScalar(0.0)) {
    return a;
  }

  switch (a.type().getKind()) {
    case Datatype::Float32:
      return ir::Call::make("atan2f", args, a.type());
    case Datatype::Float64:
      return ir::Call::make("atan2", args, a.type());
    default:
      taco_not_supported_yet;
      break;
  }
  return ir::Expr();
}

std::vector<std::vector<size_t>>
Atan2Intrinsic::zeroPreservingArgs(const std::vector<IndexExpr>& args) const {
  taco_iassert(args.size() == 2);

  IndexExpr b = args[1];

  switch (b.getDataType().getKind()) {
    case Datatype::Float32:
      if (isa<Literal>(b) && to<Literal>(b).getVal<float>() > 0.0) {
        return {{0}};
      }
      break;
    case Datatype::Float64:
      if (isa<Literal>(b) && to<Literal>(b).getVal<double>() > 0.0) {
        return {{0}};
      }
      break;
    default:
      taco_not_supported_yet;
      break;
  }

  return {{0, 1}};
}


// class SinhIntrinsic

std::string SinhIntrinsic::getName() const {
  return "sinh";
}
  
Datatype SinhIntrinsic::inferReturnType(const std::vector<Datatype>& argTypes) const {
  taco_iassert(argTypes.size() == 1);
  return argTypes[0];
}

ir::Expr SinhIntrinsic::lower(const std::vector<ir::Expr>& args) const {
  taco_iassert(args.size() == 1);

  ir::Expr arg = args[0];

  if (ir::isa<ir::Literal>(arg) && 
      (ir::to<ir::Literal>(arg)->equalsScalar(0.0))) {
    return arg;
  }

  switch (arg.type().getKind()) {
    case Datatype::Float32:
      return ir::Call::make("sinhf", args, arg.type());
    case Datatype::Float64:
      return ir::Call::make("sinh", args, arg.type());
    case Datatype::Complex64:
      return ir::Call::make("csinhf", args, arg.type());
    case Datatype::Complex128:
      return ir::Call::make("csinh", args, arg.type());
    default:
      taco_not_supported_yet;
      break;
  }
  return ir::Expr();
}

std::vector<std::vector<size_t>>
SinhIntrinsic::zeroPreservingArgs(const std::vector<IndexExpr>& args) const {
  return {{0}};
}


// class CoshIntrinsic

std::string CoshIntrinsic::getName() const {
  return "cosh";
}
  
Datatype CoshIntrinsic::inferReturnType(const std::vector<Datatype>& argTypes) const {
  taco_iassert(argTypes.size() == 1);
  return argTypes[0];
}

ir::Expr CoshIntrinsic::lower(const std::vector<ir::Expr>& args) const {
  taco_iassert(args.size() == 1);

  ir::Expr arg = args[0];
  const bool argZero = (ir::isa<ir::Literal>(arg) && 
                        ir::to<ir::Literal>(arg)->equalsScalar(0.0));

  switch (arg.type().getKind()) {
    case Datatype::Float32:
      return argZero ? ir::Literal::make((float)1.0) : 
             ir::Call::make("coshf", args, arg.type());
    case Datatype::Float64:
      return argZero ? ir::Literal::make((double)1.0) : 
             ir::Call::make("cosh", args, arg.type());
    case Datatype::Complex64:
      return argZero ? ir::Literal::make(std::complex<float>(1.0)) : 
             ir::Call::make("ccoshf", args, arg.type());
    case Datatype::Complex128:
      return argZero ? ir::Literal::make(std::complex<double>(1.0)) : 
             ir::Call::make("ccosh", args, arg.type());
    default:
      taco_not_supported_yet;
      break;
  }
  return ir::Expr();
}

std::vector<std::vector<size_t>> 
CoshIntrinsic::zeroPreservingArgs(const std::vector<IndexExpr>& args) const {
  return {};
}


// class TanhIntrinsic

std::string TanhIntrinsic::getName() const {
  return "tanh";
}
  
Datatype TanhIntrinsic::inferReturnType(const std::vector<Datatype>& argTypes) const {
  taco_iassert(argTypes.size() == 1);
  return argTypes[0];
}

ir::Expr TanhIntrinsic::lower(const std::vector<ir::Expr>& args) const {
  taco_iassert(args.size() == 1);

  ir::Expr arg = args[0];

  if (ir::isa<ir::Literal>(arg) && 
      (ir::to<ir::Literal>(arg)->equalsScalar(0.0))) {
    return arg;
  }

  switch (arg.type().getKind()) {
    case Datatype::Float32:
      return ir::Call::make("tanhf", args, arg.type());
    case Datatype::Float64:
      return ir::Call::make("tanh", args, arg.type());
    case Datatype::Complex64:
      return ir::Call::make("ctanhf", args, arg.type());
    case Datatype::Complex128:
      return ir::Call::make("ctanh", args, arg.type());
    default:
      taco_not_supported_yet;
      break;
  }
  return ir::Expr();
}

std::vector<std::vector<size_t>>
TanhIntrinsic::zeroPreservingArgs(const std::vector<IndexExpr>& args) const {
  return {{0}};
}


// class AsinhIntrinsic

std::string AsinhIntrinsic::getName() const {
  return "asinh";
}
  
Datatype AsinhIntrinsic::inferReturnType(const std::vector<Datatype>& argTypes) const {
  taco_iassert(argTypes.size() == 1);
  return argTypes[0];
}

ir::Expr AsinhIntrinsic::lower(const std::vector<ir::Expr>& args) const {
  taco_iassert(args.size() == 1);

  ir::Expr arg = args[0];

  if (ir::isa<ir::Literal>(arg) && 
      (ir::to<ir::Literal>(arg)->equalsScalar(0.0))) {
    return arg;
  }

  switch (arg.type().getKind()) {
    case Datatype::Float32:
      return ir::Call::make("asinhf", args, arg.type());
    case Datatype::Float64:
      return ir::Call::make("asinh", args, arg.type());
    case Datatype::Complex64:
      return ir::Call::make("casinhf", args, arg.type());
    case Datatype::Complex128:
      return ir::Call::make("casinh", args, arg.type());
    default:
      taco_not_supported_yet;
      break;
  }
  return ir::Expr();
}

std::vector<std::vector<size_t>>
AsinhIntrinsic::zeroPreservingArgs(const std::vector<IndexExpr>& args) const {
  return {{0}};
}


// class AcoshIntrinsic

std::string AcoshIntrinsic::getName() const {
  return "acosh";
}
  
Datatype AcoshIntrinsic::inferReturnType(const std::vector<Datatype>& argTypes) const {
  taco_iassert(argTypes.size() == 1);
  return argTypes[0];
}

ir::Expr AcoshIntrinsic::lower(const std::vector<ir::Expr>& args) const {
  taco_iassert(args.size() == 1);

  ir::Expr arg = args[0];

  switch (arg.type().getKind()) {
    case Datatype::Float32:
      return ir::Call::make("acoshf", args, arg.type());
    case Datatype::Float64:
      return ir::Call::make("acosh", args, arg.type());
    case Datatype::Complex64:
      return ir::Call::make("cacoshf", args, arg.type());
    case Datatype::Complex128:
      return ir::Call::make("cacosh", args, arg.type());
    default:
      taco_not_supported_yet;
      break;
  }
  return ir::Expr();
}

std::vector<std::vector<size_t>> 
AcoshIntrinsic::zeroPreservingArgs(const std::vector<IndexExpr>& args) const {
  return {};
}


// class AtanhIntrinsic

std::string AtanhIntrinsic::getName() const {
  return "atanh";
}
  
Datatype AtanhIntrinsic::inferReturnType(const std::vector<Datatype>& argTypes) const {
  taco_iassert(argTypes.size() == 1);
  return argTypes[0];
}

ir::Expr AtanhIntrinsic::lower(const std::vector<ir::Expr>& args) const {
  taco_iassert(args.size() == 1);

  ir::Expr arg = args[0];

  if (ir::isa<ir::Literal>(arg) && 
      (ir::to<ir::Literal>(arg)->equalsScalar(0.0))) {
    return arg;
  }

  switch (arg.type().getKind()) {
    case Datatype::Float32:
      return ir::Call::make("atanhf", args, arg.type());
    case Datatype::Float64:
      return ir::Call::make("atanh", args, arg.type());
    case Datatype::Complex64:
      return ir::Call::make("catanhf", args, arg.type());
    case Datatype::Complex128:
      return ir::Call::make("catanh", args, arg.type());
    default:
      taco_not_supported_yet;
      break;
  }
  return ir::Expr();
}

std::vector<std::vector<size_t>>
AtanhIntrinsic::zeroPreservingArgs(const std::vector<IndexExpr>& args) const {
  return {{0}};
}


// class GtIntrinsic

std::string GtIntrinsic::getName() const {
  return "gt";
}
  
Datatype GtIntrinsic::inferReturnType(const std::vector<Datatype>& argTypes) const {
  return Bool;
}

ir::Expr GtIntrinsic::lower(const std::vector<ir::Expr>& args) const {
  taco_iassert(args.size() == 2);

  ir::Expr a = args[0];
  ir::Expr b = args[1];

  return ir::Gt::make(a, b);
}

std::vector<std::vector<size_t>>
GtIntrinsic::zeroPreservingArgs(const std::vector<IndexExpr>& args) const {
  taco_iassert(args.size() == 2);

  IndexExpr a = args[0];
  IndexExpr b = args[1];

  switch (b.getDataType().getKind()) {
    case Datatype::UInt8:
    case Datatype::UInt16:
    case Datatype::UInt32:
    case Datatype::UInt64:
      return {{0}};
    case Datatype::Int8: 
      if (isa<Literal>(b) && to<Literal>(b).getVal<int8_t>() >= 0) {
        return {{0}};
      } else if (isa<Literal>(a) && to<Literal>(a).getVal<int8_t>() <= 0) {
        return {{1}};
      }
      break;
    case Datatype::Int16:
      if (isa<Literal>(b) && to<Literal>(b).getVal<int16_t>() >= 0) {
        return {{0}};
      } else if (isa<Literal>(a) && to<Literal>(a).getVal<int16_t>() <= 0) {
        return {{1}};
      }
      break;
    case Datatype::Int32:
      if (isa<Literal>(b) && to<Literal>(b).getVal<int32_t>() >= 0) {
        return {{0}};
      } else if (isa<Literal>(a) && to<Literal>(a).getVal<int32_t>() <= 0) {
        return {{1}};
      }
      break;
    case Datatype::Int64:
      if (isa<Literal>(b) && to<Literal>(b).getVal<int64_t>() >= 0) {
        return {{0}};
      } else if (isa<Literal>(a) && to<Literal>(a).getVal<int64_t>() <= 0) {
        return {{1}};
      }
      break;
    case Datatype::Float32:
      if (isa<Literal>(b) && to<Literal>(b).getVal<float>() >= 0.0) {
        return {{0}};
      } else if (isa<Literal>(a) && to<Literal>(a).getVal<float>() <= 0.0) {
        return {{1}};
      }
      break;
    case Datatype::Float64:
      if (isa<Literal>(b) && to<Literal>(b).getVal<double>() >= 0.0) {
        return {{0}};
      } else if (isa<Literal>(a) && to<Literal>(a).getVal<double>() <= 0.0) {
        return {{1}};
      }
      break;
    default:
      taco_not_supported_yet;
      break;
  }

  return {{0, 1}};
}


// class LtIntrinsic

std::string LtIntrinsic::getName() const {
  return "lt";
}
  
Datatype LtIntrinsic::inferReturnType(const std::vector<Datatype>& argTypes) const {
  return Bool;
}

ir::Expr LtIntrinsic::lower(const std::vector<ir::Expr>& args) const {
  taco_iassert(args.size() == 2);

  ir::Expr a = args[0];
  ir::Expr b = args[1];

  return ir::Lt::make(a, b);
}

std::vector<std::vector<size_t>>
LtIntrinsic::zeroPreservingArgs(const std::vector<IndexExpr>& args) const {
  taco_iassert(args.size() == 2);

  IndexExpr a = args[0];
  IndexExpr b = args[1];

  switch (b.getDataType().getKind()) {
    case Datatype::UInt8:
    case Datatype::UInt16:
    case Datatype::UInt32:
    case Datatype::UInt64:
      return {{1}};
    case Datatype::Int8: 
      if (isa<Literal>(a) && to<Literal>(a).getVal<int8_t>() >= 0) {
        return {{1}};
      } else if (isa<Literal>(b) && to<Literal>(b).getVal<int8_t>() <= 0) {
        return {{0}};
      }
      break;
    case Datatype::Int16:
      if (isa<Literal>(a) && to<Literal>(a).getVal<int16_t>() >= 0) {
        return {{1}};
      } else if (isa<Literal>(b) && to<Literal>(b).getVal<int16_t>() <= 0) {
        return {{0}};
      }
      break;
    case Datatype::Int32:
      if (isa<Literal>(a) && to<Literal>(a).getVal<int32_t>() >= 0) {
        return {{1}};
      } else if (isa<Literal>(b) && to<Literal>(b).getVal<int32_t>() <= 0) {
        return {{0}};
      }
      break;
    case Datatype::Int64:
      if (isa<Literal>(a) && to<Literal>(a).getVal<int64_t>() >= 0) {
        return {{1}};
      } else if (isa<Literal>(b) && to<Literal>(b).getVal<int64_t>() <= 0) {
        return {{0}};
      }
      break;
    case Datatype::Float32:
      if (isa<Literal>(a) && to<Literal>(a).getVal<float>() >= 0.0) {
        return {{1}};
      } else if (isa<Literal>(b) && to<Literal>(b).getVal<float>() <= 0.0) {
        return {{0}};
      }
      break;
    case Datatype::Float64:
      if (isa<Literal>(a) && to<Literal>(a).getVal<double>() >= 0.0) {
        return {{1}};
      } else if (isa<Literal>(b) && to<Literal>(b).getVal<double>() <= 0.0) {
        return {{0}};
      }
      break;
    default:
      taco_not_supported_yet;
      break;
  }

  return {{0, 1}};
}


// class GteIntrinsic

std::string GteIntrinsic::getName() const {
  return "gte";
}
  
Datatype GteIntrinsic::inferReturnType(const std::vector<Datatype>& argTypes) const {
  return Bool;
}

ir::Expr GteIntrinsic::lower(const std::vector<ir::Expr>& args) const {
  taco_iassert(args.size() == 2);

  ir::Expr a = args[0];
  ir::Expr b = args[1];

  return ir::Gte::make(a, b);
}

std::vector<std::vector<size_t>>
GteIntrinsic::zeroPreservingArgs(const std::vector<IndexExpr>& args) const {
  taco_iassert(args.size() == 2);

  IndexExpr a = args[0];
  IndexExpr b = args[1];

  switch (b.getDataType().getKind()) {
    case Datatype::UInt8:
      if (isa<Literal>(b) && to<Literal>(b).getVal<uint8_t>() > 0) {
        return {{0}};
      }
      break;
    case Datatype::UInt16:
      if (isa<Literal>(b) && to<Literal>(b).getVal<uint16_t>() > 0) {
        return {{0}};
      }
      break;
    case Datatype::UInt32:
      if (isa<Literal>(b) && to<Literal>(b).getVal<uint32_t>() > 0) {
        return {{0}};
      }
      break;
    case Datatype::UInt64:
      if (isa<Literal>(b) && to<Literal>(b).getVal<uint64_t>() > 0) {
        return {{0}};
      }
      break;
    case Datatype::Int8: 
      if (isa<Literal>(b) && to<Literal>(b).getVal<int8_t>() > 0) {
        return {{0}};
      } else if (isa<Literal>(a) && to<Literal>(a).getVal<int8_t>() < 0) {
        return {{1}};
      }
      break;
    case Datatype::Int16:
      if (isa<Literal>(b) && to<Literal>(b).getVal<int16_t>() > 0) {
        return {{0}};
      } else if (isa<Literal>(a) && to<Literal>(a).getVal<int16_t>() < 0) {
        return {{1}};
      }
      break;
    case Datatype::Int32:
      if (isa<Literal>(b) && to<Literal>(b).getVal<int32_t>() > 0) {
        return {{0}};
      } else if (isa<Literal>(a) && to<Literal>(a).getVal<int32_t>() < 0) {
        return {{1}};
      }
      break;
    case Datatype::Int64:
      if (isa<Literal>(b) && to<Literal>(b).getVal<int64_t>() > 0) {
        return {{0}};
      } else if (isa<Literal>(a) && to<Literal>(a).getVal<int64_t>() < 0) {
        return {{1}};
      }
      break;
    case Datatype::Float32:
      if (isa<Literal>(b) && to<Literal>(b).getVal<float>() > 0.0) {
        return {{0}};
      } else if (isa<Literal>(a) && to<Literal>(a).getVal<float>() < 0.0) {
        return {{1}};
      }
      break;
    case Datatype::Float64:
      if (isa<Literal>(b) && to<Literal>(b).getVal<double>() > 0.0) {
        return {{0}};
      } else if (isa<Literal>(a) && to<Literal>(a).getVal<double>() < 0.0) {
        return {{1}};
      }
      break;
    default:
      taco_not_supported_yet;
      break;
  }

  return {};
}


// class LteIntrinsic

std::string LteIntrinsic::getName() const {
  return "lte";
}
  
Datatype LteIntrinsic::inferReturnType(const std::vector<Datatype>& argTypes) const {
  return Bool;
}

ir::Expr LteIntrinsic::lower(const std::vector<ir::Expr>& args) const {
  taco_iassert(args.size() == 2);

  ir::Expr a = args[0];
  ir::Expr b = args[1];

  return ir::Lte::make(a, b);
}

std::vector<std::vector<size_t>>
LteIntrinsic::zeroPreservingArgs(const std::vector<IndexExpr>& args) const {
  taco_iassert(args.size() == 2);

  IndexExpr a = args[0];
  IndexExpr b = args[1];

  switch (b.getDataType().getKind()) {
    case Datatype::UInt8:
      if (isa<Literal>(a) && to<Literal>(a).getVal<uint8_t>() > 0) {
        return {{1}};
      }
      break;
    case Datatype::UInt16:
      if (isa<Literal>(a) && to<Literal>(a).getVal<uint16_t>() > 0) {
        return {{1}};
      }
      break;
    case Datatype::UInt32:
      if (isa<Literal>(a) && to<Literal>(a).getVal<uint32_t>() > 0) {
        return {{1}};
      }
      break;
    case Datatype::UInt64:
      if (isa<Literal>(a) && to<Literal>(a).getVal<uint64_t>() > 0) {
        return {{1}};
      }
      break;
    case Datatype::Int8: 
      if (isa<Literal>(a) && to<Literal>(a).getVal<int8_t>() > 0) {
        return {{1}};
      } else if (isa<Literal>(b) && to<Literal>(b).getVal<int8_t>() < 0) {
        return {{0}};
      }
      break;
    case Datatype::Int16:
      if (isa<Literal>(a) && to<Literal>(a).getVal<int16_t>() > 0) {
        return {{1}};
      } else if (isa<Literal>(b) && to<Literal>(b).getVal<int16_t>() < 0) {
        return {{0}};
      }
      break;
    case Datatype::Int32:
      if (isa<Literal>(a) && to<Literal>(a).getVal<int32_t>() > 0) {
        return {{1}};
      } else if (isa<Literal>(b) && to<Literal>(b).getVal<int32_t>() < 0) {
        return {{0}};
      }
      break;
    case Datatype::Int64:
      if (isa<Literal>(a) && to<Literal>(a).getVal<int64_t>() > 0) {
        return {{1}};
      } else if (isa<Literal>(b) && to<Literal>(b).getVal<int64_t>() < 0) {
        return {{0}};
      }
      break;
    case Datatype::Float32:
      if (isa<Literal>(a) && to<Literal>(a).getVal<float>() > 0.0) {
        return {{1}};
      } else if (isa<Literal>(b) && to<Literal>(b).getVal<float>() < 0.0) {
        return {{0}};
      }
      break;
    case Datatype::Float64:
      if (isa<Literal>(a) && to<Literal>(a).getVal<double>() > 0.0) {
        return {{1}};
      } else if (isa<Literal>(b) && to<Literal>(b).getVal<double>() < 0.0) {
        return {{0}};
      }
      break;
    default:
      taco_not_supported_yet;
      break;
  }

  return {};
}


// class EqIntrinsic

std::string EqIntrinsic::getName() const {
  return "eq";
}
  
Datatype EqIntrinsic::inferReturnType(const std::vector<Datatype>& argTypes) const {
  return Bool;
}

ir::Expr EqIntrinsic::lower(const std::vector<ir::Expr>& args) const {
  taco_iassert(args.size() == 2);

  ir::Expr a = args[0];
  ir::Expr b = args[1];

  return ir::Eq::make(a, b);
}

std::vector<std::vector<size_t>> 
EqIntrinsic::zeroPreservingArgs(const std::vector<IndexExpr>& args) const {
  return {};
}


// class NeqIntrinsic

std::string NeqIntrinsic::getName() const {
  return "neq";
}
  
Datatype NeqIntrinsic::inferReturnType(const std::vector<Datatype>& argTypes) const {
  return Bool;
}

ir::Expr NeqIntrinsic::lower(const std::vector<ir::Expr>& args) const {
  taco_iassert(args.size() == 2);

  ir::Expr a = args[0];
  ir::Expr b = args[1];

  return ir::Neq::make(a, b);
}

std::vector<std::vector<size_t>> 
NeqIntrinsic::zeroPreservingArgs(const std::vector<IndexExpr>& args) const {
  if (equals(args[0], Literal::zero(args[0].getDataType()))) {
    return {{1}};
  } else if (equals(args[1], Literal::zero(args[1].getDataType()))) {
    return {{0}};
  }

  return {{0, 1}};
}


// class MaxIntrinsic

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
    return a;
  }

  return ir::Max::make(a, b);
}

std::vector<std::vector<size_t>>
MaxIntrinsic::zeroPreservingArgs(const std::vector<IndexExpr>& args) const {
  taco_iassert(args.size() == 2);

  if (equals(args[0], Literal::zero(args[0].getDataType()))) {
    return {{1}};
  } else if (equals(args[1], Literal::zero(args[1].getDataType()))) {
    return {{0}};
  }

  return {{0, 1}};
}


// class MinIntrinsic

std::string MinIntrinsic::getName() const {
  return "max";
}
  
Datatype MinIntrinsic::inferReturnType(const std::vector<Datatype>& argTypes) const {
  taco_iassert(argTypes.size() == 2);
  taco_iassert(argTypes[0] == argTypes[1]);
  return argTypes[0];
}

ir::Expr MinIntrinsic::lower(const std::vector<ir::Expr>& args) const {
  taco_iassert(args.size() == 2);

  ir::Expr a = args[0];
  ir::Expr b = args[1];

  if (ir::isa<ir::Literal>(a) && 
      ir::to<ir::Literal>(a)->equalsScalar(0.0) &&
      ir::isa<ir::Literal>(b) &&
      ir::to<ir::Literal>(b)->equalsScalar(0.0)) {
    return a;
  }

  return ir::Min::make(a, b);
}

std::vector<std::vector<size_t>>
MinIntrinsic::zeroPreservingArgs(const std::vector<IndexExpr>& args) const {
  taco_iassert(args.size() == 2);

  if (equals(args[0], Literal::zero(args[0].getDataType()))) {
    return {{1}};
  } else if (equals(args[1], Literal::zero(args[1].getDataType()))) {
    return {{0}};
  }

  return {{0, 1}};
}


// class HeavisideIntrinsic

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

std::vector<std::vector<size_t>>
HeavisideIntrinsic::zeroPreservingArgs(const std::vector<IndexExpr>& args) const {
  taco_iassert(args.size() == 2);

  if (equals(args[1], Literal::zero(args[1].getDataType()))) {
    return {{0}};
  }

  return {};
}


// class NotIntrinsic

std::string NotIntrinsic::getName() const {
  return "Not";
}
  
Datatype NotIntrinsic::inferReturnType(const std::vector<Datatype>& argTypes) const {
  return Bool;
}

ir::Expr NotIntrinsic::lower(const std::vector<ir::Expr>& args) const {
  taco_iassert(args.size() == 1);

  ir::Expr a = args[0];

  if (ir::isa<ir::Literal>(a) && 
      ir::to<ir::Literal>(a)->equalsScalar(0)) {
    return ir::Literal::make(true);
  }

  return ir::Eq::make(a, ir::Literal::make(false));
}

std::vector<std::vector<size_t>>
NotIntrinsic::zeroPreservingArgs(const std::vector<IndexExpr>& args) const {
  return {};
}

}
