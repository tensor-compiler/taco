#ifndef TACO_CODEGEN_LLVM_H
#define TACO_CODEGEN_LLVM_H

#include <map>
#include <vector>

#include "taco/ir/ir.h"
#include "taco/ir/ir_visitor.h"
#include "taco/target.h"
#include "taco/util/scopedmap.h"

namespace llvm {
class Module;
class LLVMContext;
class Function;
template<typename, typename> class IRBuilder;
class Value;
class IRBuilderDefaultInserter;
class ConstantFolder;
class Type;
class StructType;
}

namespace taco {
namespace ir {

class CodeGen_LLVM : IRVisitorStrict {
public:

  /// Initialize a code generator that generates code to an
  /// output stream.
  CodeGen_LLVM(const Target &target,
               llvm::LLVMContext &context);
  ~CodeGen_LLVM();

  /// Compile a lowered function
  void compile(Stmt stmt, bool isFirst=false);

  // TODO: Remove & use name generator from IRPrinter
  static std::string genUniqueName(std::string varName="");
  
//  /// Generate shims that unpack an array of pointers representing
//  /// a mix of taco_tensor_t* and scalars into a function call
//  static void generateShim(const Stmt& func, std::stringstream &stream);

  llvm::Value* getSymbol(const std::string &name);
  bool containsSymbol(const std::string &name);
  void pushSymbol(const std::string &name, llvm::Value *value);
  void pushScope();
  void popScope();
  
protected:
  // The taco target for this module
  Target target;

  // Symbol table
  util::ScopedMap<const std::string, llvm::Value*> symbolTable;

  // State needed for LLVM code generation
  static bool LLVMInitialized;
  std::unique_ptr<llvm::Module> module;        // current LLVM module
  llvm::Function *function;                    // current LLVM function
  llvm::LLVMContext *context;                  // LLVM context
  llvm::IRBuilder<llvm::ConstantFolder,
    llvm::IRBuilderDefaultInserter> *builder;  // builder for code generation
  llvm::Value *value;                          // last generated LLVM value

  // Initialization
  void init_context();

  // Emit code for a statement
  void codegen(Stmt);
  
  // Emit code for an expression
  llvm::Value *codegen(Expr);

  using IRVisitorStrict::visit;

  void visit(const Literal*);
  void visit(const Var*);
  void visit(const Neg*);
  void visit(const Sqrt*);
  void visit(const Add*);
  void visit(const Sub*);
  void visit(const Mul*);
  void visit(const Div*);
  void visit(const Rem*);
  void visit(const Min*);
  void visit(const Max*);
  void visit(const BitAnd*);
  void visit(const BitOr*);
  void visit(const Eq*);
  void visit(const Neq*);
  void visit(const Gt*);
  void visit(const Lt*);
  void visit(const Gte*);
  void visit(const Lte*);
  void visit(const And*);
  void visit(const Or*);
  void visit(const Cast*);
  void visit(const IfThenElse*);
  void visit(const Case*);
  void visit(const Switch*);
  void visit(const Load*);
  void visit(const Store*);
  void visit(const For*);
  void visit(const While*);
  void visit(const Block*);
  void visit(const Scope*);
  void visit(const Function*);
  void visit(const VarAssign*);
  void visit(const Allocate*);
  void visit(const Comment*);
  void visit(const BlankLine*);
  void visit(const Print*);
  void visit(const GetProperty*);

  // helpers
  void beginFunc(const Function *);
  void endFunc(const Function *);
  
  std::vector<Expr> currentFunctionArgs;
  std::map<Expr, std::string, ExprCompare> varMap;

  // useful types
  llvm::Type *orderType, *dimensionsType, *csizeType, *mode_orderingType,
             *mode_typesType, *indicesType, *valsType, *vals_sizeType;
  
  llvm::StructType *tacoTensorType;
  
  
};

} // namespace ir
} // namespace taco
#endif
