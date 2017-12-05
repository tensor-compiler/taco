#ifndef TACO_BACKEND_C_H
#define TACO_BACKEND_C_H

#include <map>
#include <vector>

#include "taco/ir/ir.h"
#include "taco/ir/ir_printer.h"

namespace taco {
namespace ir {


class CodeGen_C : public IRPrinter {
public:
  /// Kind of output: header or implementation
  enum OutputKind { C99Header, C99Implementation };

  /// Initialize a code generator that generates code to an
  /// output stream.
  CodeGen_C(std::ostream &dest, OutputKind outputKind);
  ~CodeGen_C();

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
  void visit(const Allocate*);
  void visit(const Sqrt*);

  std::map<Expr, std::string, ExprCompare> varMap;
  std::ostream &out;
  
  OutputKind outputKind;
};

} // namespace ir
} // namespace taco
#endif
