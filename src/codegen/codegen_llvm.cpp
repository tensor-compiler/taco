#include "codegen/codegen_llvm.h"
#include "codegen/llvm_headers.h"

namespace taco {
namespace ir {

using namespace llvm;

bool CodeGen_LLVM::LLVMInitialized = false;

CodeGen_LLVM::CodeGen_LLVM(const Target &target,
               llvm::LLVMContext &context) :
  target(target),
  function(nullptr),
  context(&context),
  builder(nullptr),
  value(nullptr) {
  if (!LLVMInitialized) {
    module = make_unique<llvm::Module>("taco module", context);
  }
}
CodeGen_LLVM::~CodeGen_LLVM() { }

  
void CodeGen_LLVM::compile(Stmt stmt, bool isFirst) {
  codegen(stmt);
}

void CodeGen_LLVM::codegen(Stmt stmt) {
  value = nullptr;
  stmt.accept(this);
}

Value *CodeGen_LLVM::codegen(Expr expr) {
  value = nullptr;
  expr.accept(this);
  taco_iassert(value) << "Codegen of expression " << expr <<
    " did not produce an LLVM value";
  return value;
}

llvm::Value* CodeGen_LLVM::getSymbol(const std::string &name) {
  return symbolTable.get(name);
}

void CodeGen_LLVM::pushSymbol(const std::string &name, llvm::Value *value) {
  symbolTable.insert({name, value});
}

namespace {

llvm::Type *llvmTypeOf(LLVMContext *context, Datatype t) {
  taco_tassert(!t.isComplex()) << "LLVM codegen for complex not yet supported";
  
  if (t.isFloat()) {
    switch (t.getNumBits()) {
      case 32:
        return llvm::Type::getFloatTy(*context);
      case 64:
        return llvm::Type::getDoubleTy(*context);
      default:
        taco_ierror << "Unabe to find LLVM type for " << t;
        return nullptr;
    }
  } else {
    return llvm::Type::getIntNTy(*context, t.getNumBits());
  }
}

} // anonymous namespace


void CodeGen_LLVM::visit(const Literal* e) {
  if (e->type.isFloat()) {
    value = ConstantFP::get(llvmTypeOf(context, e->type), e->float_value);
  } else if (e->type.isUInt()) {
    value = ConstantInt::get(llvmTypeOf(context, e->type), e->uint_value);
  } else if (e->type.isInt()) {
    value = ConstantInt::getSigned(llvmTypeOf(context, e->type), e->int_value);
  } else {
    taco_ierror << "Unable to generate LLVM for literal " << e;
  }
}

void CodeGen_LLVM::visit(const Var* e) {
  value = getSymbol(e->name);
}

void CodeGen_LLVM::visit(const Neg* e) {
  if (e->type.isFloat()) {
    value = builder->CreateFSub(Constant::getNullValue(llvmTypeOf(context, e->type)),
                                codegen(e));
  } else {
    value = builder->CreateSub(Constant::getNullValue(llvmTypeOf(context, e->type)),
                               codegen(e));
  }
}

void CodeGen_LLVM::visit(const Sqrt*) { }

void CodeGen_LLVM::visit(const Add* e) {
  if (e->type.isFloat()) {
    value = builder->CreateFAdd(codegen(e->a), codegen(e->b));
  } else {
    value = builder->CreateAdd(codegen(e->a), codegen(e->b));
  }
}

void CodeGen_LLVM::visit(const Sub* e) {
  if (e->type.isFloat()) {
    value = builder->CreateFSub(codegen(e->a), codegen(e->b));
  } else {
    value = builder->CreateSub(codegen(e->a), codegen(e->b));
  }
}

void CodeGen_LLVM::visit(const Mul* e) {
  if (e->type.isFloat()) {
    value = builder->CreateFMul(codegen(e->a), codegen(e->b));
  } else {
    value = builder->CreateMul(codegen(e->a), codegen(e->b));
  }
}

void CodeGen_LLVM::visit(const Div* e) {
  // TODO: Turning integer division into shifts/etc can sometimes be
  // fruitful.  We should implement the same ops as Halide.
  if (e->type.isFloat()) {
    value = builder->CreateFDiv(codegen(e->a), codegen(e->b));
  } else if (e->type.isUInt()){
    value = builder->CreateExactUDiv(codegen(e->a), codegen(e->b));
  } else {
    value = builder->CreateExactSDiv(codegen(e->a), codegen(e->b));
  }
}

void CodeGen_LLVM::visit(const Rem*) { }
void CodeGen_LLVM::visit(const Min*) { }
void CodeGen_LLVM::visit(const Max*) { }
void CodeGen_LLVM::visit(const BitAnd*) { }
void CodeGen_LLVM::visit(const BitOr*) { }
void CodeGen_LLVM::visit(const Eq*) { }
void CodeGen_LLVM::visit(const Neq*) { }
void CodeGen_LLVM::visit(const Gt*) { }
void CodeGen_LLVM::visit(const Lt*) { }
void CodeGen_LLVM::visit(const Gte*) { }
void CodeGen_LLVM::visit(const Lte*) { }
void CodeGen_LLVM::visit(const And*) { }
void CodeGen_LLVM::visit(const Or*) { }
void CodeGen_LLVM::visit(const Cast*) { }
void CodeGen_LLVM::visit(const IfThenElse*) { }
void CodeGen_LLVM::visit(const Case*) { }
void CodeGen_LLVM::visit(const Switch*) { }
void CodeGen_LLVM::visit(const Load*) { }
void CodeGen_LLVM::visit(const Store*) { }
void CodeGen_LLVM::visit(const For*) { }
void CodeGen_LLVM::visit(const While*) { }
void CodeGen_LLVM::visit(const Block*) { }
void CodeGen_LLVM::visit(const Scope*) { }
void CodeGen_LLVM::visit(const Function*) { }
void CodeGen_LLVM::visit(const VarAssign*) { }
void CodeGen_LLVM::visit(const Allocate*) { }
void CodeGen_LLVM::visit(const Comment*) { }
void CodeGen_LLVM::visit(const BlankLine*) { }
void CodeGen_LLVM::visit(const Print*) { }
void CodeGen_LLVM::visit(const GetProperty*) { }

} // namespace ir
} // namespace taco
