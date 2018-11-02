#ifndef TACO_BACKEND_CUDA_H
#define TACO_BACKEND_CUDA_H

#include <map>
#include <vector>

#include "taco/ir/ir.h"
#include "taco/ir/ir_printer.h"

namespace taco {
namespace ir {


class CodeGen_CUDA : public IRPrinter {
public:
  /// Kind of output: header or implementation
  enum OutputKind { C99Header, C99Implementation };

  /// Initialize a code generator that generates code to an
  /// output stream.
  CodeGen_CUDA(std::ostream &dest, OutputKind outputKind);
  ~CodeGen_CUDA();

  /// Compile a lowered function
  void compile(Stmt stmt, bool isFirst=false);

  // TODO: Remove & use name generator from IRPrinter
  static std::string genUniqueName(std::string varName="");
  
  /// Generate shims that unpack an array of pointers representing
  /// a mix of taco_tensor_t* and scalars into a function call
  static void generateShim(const Stmt& func, std::stringstream &stream);
  
protected:
  using IRPrinter::visit;
  void visit(const Function*);
  void visit(const Var*);
  void visit(const For*);
  void visit(const While*);
  void visit(const GetProperty*);
  void visit(const Min*);
  void visit(const Max*);
  void visit(const Allocate*);
  void visit(const Sqrt*);
  static std::string printDeviceFuncName(const std::vector<std::pair<std::string, Expr>> currentParameters, int index);
  void printDeviceFuncCall(const std::vector<std::pair<std::string, Expr>> currentParameters, int index, Expr start, Expr end);
  void printThreadIDVariable(std::pair<std::string, Expr> threadIDVar);
  void printDeviceFunctions(const Function* func);
  std::map<Expr, std::string, ExprCompare> varMap;

  std::vector<std::vector<std::pair<std::string, Expr>>> deviceFunctionParameters;
  std::vector<Stmt> deviceFunctions; // expressions to replace to calls of device function

  std::ostream &out;
  
  OutputKind outputKind;
};

} // namespace ir
} // namespace taco
#endif
