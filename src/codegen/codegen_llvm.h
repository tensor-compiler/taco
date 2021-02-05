#ifndef TACO_BACKEND_LLVM_H
#define TACO_BACKEND_LLVM_H

#include "codegen.h"
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>

namespace taco {
namespace ir {

class CodeGen_LLVM : public CodeGen {
private:
  // LLVM stuff
  llvm::IRBuilder<> *Builder = nullptr;
  llvm::LLVMContext Context;
  llvm::Function *F = nullptr;

  // Symbol Table
  util::ScopedMap<const std::string, llvm::Value *> symbolTable;

  OutputKind outputKind;
  class FindVars;

  llvm::StructType *tensorType;
  llvm::Value *value; // last llvm value generated

public:
  CodeGen_LLVM(std::ostream &stream, OutputKind kind)
      : CodeGen(stream, LLVM), outputKind(kind){};
  CodeGen_LLVM(std::ostream &stream, bool color, bool simplify, OutputKind kind)
      : CodeGen(stream, color, simplify, LLVM), outputKind(kind){};
  void compile(Stmt stmt, bool isFirst = false) override;

protected:
  using IRPrinter::visit;

  // symbol table related
  void pushSymbol(const std::string &name, llvm::Value *v);
  llvm::Value *getSymbol(const std::string &name);
  void pushScope();
  void popScope();

  // utils
  void init_codegen();
  llvm::Type *llvmTypeOf(Datatype);

  void codegen(const Stmt);
  llvm::Value *codegen(const Expr);

  void visit(const Literal *) override;
  void visit(const Var *) override;
  void visit(const Neg *) override;
  void visit(const Sqrt *) override;
  void visit(const Add *) override;
  void visit(const Sub *) override;
  void visit(const Mul *) override;
  void visit(const Div *) override;
  void visit(const Rem *) override;
  void visit(const Min *) override;
  void visit(const Max *) override;
  void visit(const BitAnd *) override;
  void visit(const BitOr *) override;
  void visit(const Eq *) override;
  void visit(const Neq *) override;
  void visit(const Gt *) override;
  void visit(const Lt *) override;
  void visit(const Gte *) override;
  void visit(const Lte *) override;
  void visit(const And *) override;
  void visit(const Or *) override;
  void visit(const Cast *) override;
  void visit(const Call *) override;
  void visit(const IfThenElse *) override;
  void visit(const Case *) override;
  void visit(const Switch *) override;
  void visit(const Load *) override;
  void visit(const Malloc *) override;
  void visit(const Sizeof *) override;
  void visit(const Store *) override;
  void visit(const For *) override;
  void visit(const While *) override;
  void visit(const Block *) override;
  void visit(const Scope *) override;
  void visit(const Function *) override;
  void visit(const VarDecl *) override;
  void visit(const Assign *) override;
  void visit(const Yield *) override;
  void visit(const Allocate *) override;
  void visit(const Free *) override;
  void visit(const Comment *) override;
  void visit(const BlankLine *) override;
  void visit(const Break *) override;
  void visit(const Print *) override;
  void visit(const GetProperty *) override;
};

} // namespace ir
} // namespace taco

#endif