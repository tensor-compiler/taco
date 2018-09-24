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
  void visit(const Max*);
  void visit(const Allocate*);
  void visit(const Sqrt*);

  std::map<Expr, std::string, ExprCompare> varMap;
  std::ostream &out;
  
  OutputKind outputKind;
  
  // find variables for generating declarations
  // also only generates a single var for each GetProperty
  class FindVars : public IRVisitor {
  public:
    FindVars(std::vector<Expr> inputs, std::vector<Expr> outputs);
    
    std::map<Expr, std::string, ExprCompare> varMap;
  
    // the variables for which we need to add declarations
    std::map<Expr, std::string, ExprCompare> varDecls;
  
    // this maps from tensor, property, mode, index to the unique var
    std::map<std::tuple<Expr, TensorProperty, int, int>, std::string> canonicalPropertyVar;
  
    // this is for convenience, recording just the properties unpacked
    // from the output tensor so we can re-save them at the end
    std::map<std::tuple<Expr, TensorProperty, int, int>, std::string> outputProperties;
  
    // TODO: should replace this with an unordered set
    std::vector<Expr> outputTensors;
  protected:
    bool inVarAssignLHSWithDecl;
    using IRVisitor::visit;
    
    virtual void visit(const For *op);
    virtual void visit(const Var *op);
    virtual void visit(const VarDecl *op);
    virtual void visit(const GetProperty *op);
  };
};

} // namespace ir
} // namespace taco
#endif
