#ifndef TACO_BACKEND_LLVM_H
#define TACO_BACKEND_LLVM_H

#include "codegen.h"
#include <llvm/IR/LLVMContext.h>

namespace taco{
namespace ir{

class CodeGen_LLVM : public CodeGen {
private:
  llvm::LLVMContext llvmContext;
  OutputKind outputKind;
  // protected:
public:

  CodeGen_LLVM(std::ostream& stream, OutputKind kind) : CodeGen(stream, LLVM), outputKind(kind) { };
  CodeGen_LLVM(std::ostream& stream, bool color, bool simplify, OutputKind kind) : CodeGen(stream, color, simplify, LLVM), outputKind(kind) { };
  void compile(Stmt stmt, bool isFirst=false) override;
};

} // namespace ir
} // namespace taco

#endif