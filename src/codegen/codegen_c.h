#ifndef TACO_BACKEND_C_H
#define TACO_BACKEND_C_H
#include <map>
#include <vector>

#include "taco/ir/ir.h"
#include "taco/ir/ir_printer.h"
#include "codegen.h"

namespace taco {
namespace ir {


class CodeGen_C : public CodeGen {
public:
  /// Initialize a code generator that generates code to an
  /// output stream.
  CodeGen_C(std::ostream &dest, OutputKind outputKind, bool simplify=true);
  ~CodeGen_C();

  /// Compile a lowered function
  void compile(Stmt stmt, bool isFirst=false);

  /// Generate shims that unpack an array of pointers representing
  /// a mix of taco_tensor_t* and scalars into a function call
  static void generateShim(const Stmt& func, std::stringstream &stream);

protected:
  using IRPrinter::visit;

  void visit(const Function*);
  void visit(const VarDecl*);
  void visit(const Yield*);
  void visit(const Var*);
  void visit(const For*);
  void visit(const While*);
  void visit(const GetProperty*);
  void visit(const Min*);
  void visit(const Max*);
  void visit(const Allocate*);
  void visit(const Sqrt*);
  void visit(const Store*);
  void visit(const Assign*);

  std::map<Expr, std::string, ExprCompare> varMap;
  std::vector<Expr> localVars;
  std::ostream &out;
  
  OutputKind outputKind;

  std::string funcName;
  int labelCount;
  bool emittingCoroutine;

  class FindVars;

private:
  virtual std::string restrictKeyword() const { return "restrict"; }
};

} // namespace ir
} // namespace taco
#endif
