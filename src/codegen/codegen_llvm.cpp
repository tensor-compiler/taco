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

void CodeGen_LLVM::visit(const Literal*) { }
void CodeGen_LLVM::visit(const Var*) { }
void CodeGen_LLVM::visit(const Neg*) { }
void CodeGen_LLVM::visit(const Sqrt*) { }
void CodeGen_LLVM::visit(const Add*) { }
void CodeGen_LLVM::visit(const Sub*) { }
void CodeGen_LLVM::visit(const Mul*) { }
void CodeGen_LLVM::visit(const Div*) { }
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
