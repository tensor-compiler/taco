#ifndef TACO_BACKEND_LLVM_H
#define TACO_BACKEND_LLVM_H

#include "codegen.h"
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/GlobalValue.h>


namespace taco{
namespace ir{

class CodeGen_LLVM : public CodeGen {
private:
  // LLVM stuff
  llvm::IRBuilder<> *Builder = nullptr;
  llvm::LLVMContext Context;

  // Symbol Table
  util::ScopedMap<const std::string, llvm::Value*> symbolTable;

  OutputKind outputKind;
  class FindVars;

  llvm::StructType *tensorType;
  llvm::Value *value; // last llvm value generated

public:
  CodeGen_LLVM(std::ostream& stream, OutputKind kind) : CodeGen(stream, LLVM), outputKind(kind) { };
  CodeGen_LLVM(std::ostream& stream, bool color, bool simplify, OutputKind kind) : CodeGen(stream, color, simplify, LLVM), outputKind(kind) { };
  void compile(Stmt stmt, bool isFirst=false) override;

protected:
  using IRPrinter::visit;

  // symbol table related
  void pushSymbol(const std::string &name, llvm::Value *v);
  llvm::Value* getSymbol(const std::string &name);
  void pushScope();
  void popScope();

  // utils
  void init_codegen();
  llvm::Type* llvmTypeOf(Datatype);

  void codegen(const Stmt);
  llvm::Value* codegen(const Expr);

  void visit(const Literal *);
  void visit(const Var *);
  void visit(const Neg *);
  void visit(const Sqrt *);
  void visit(const Add *);
  void visit(const Sub *);
  void visit(const Mul *);
  void visit(const Div *);
  void visit(const Rem *);
  void visit(const Min *);
  void visit(const Max *);
  void visit(const BitAnd *);
  void visit(const BitOr *);
  void visit(const Eq *);
  void visit(const Neq *);
  void visit(const Gt *);
  void visit(const Lt *);
  void visit(const Gte *);
  void visit(const Lte *);
  void visit(const And *);
  void visit(const Or *);
  void visit(const Cast *);
  void visit(const Call *);
  void visit(const IfThenElse *);
  void visit(const Case *);
  void visit(const Switch *);
  void visit(const Load *);
  void visit(const Malloc *);
  void visit(const Sizeof *);
  void visit(const Store *);
  void visit(const For *);
  void visit(const While *);
  void visit(const Block *);
  void visit(const Scope *);
  void visit(const Function *);
  void visit(const VarDecl *);
  void visit(const Assign *);
  void visit(const Yield *);
  void visit(const Allocate *);
  void visit(const Free *);
  void visit(const Comment *);
  void visit(const BlankLine *);
  void visit(const Break *);
  void visit(const Print *);
  void visit(const GetProperty *);
};

} // namespace ir
} // namespace taco

#endif