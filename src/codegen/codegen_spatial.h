#ifndef TACO_BACKEND_SPATIAL_H
#define TACO_BACKEND_SPATIAL_H
#include <map>
#include <vector>

#include "taco/ir/ir.h"
#include "taco/ir/ir_printer.h"
//#include "taco/ir/ir_printer_spatial.h"
#include "codegen.h"

namespace taco {
namespace ir {


class CodeGen_Spatial : public CodeGen {
public:
  /// Initialize a code generator that generates code to an
  /// output stream.
  CodeGen_Spatial(std::ostream &dest, OutputKind outputKind, bool simplify=true);
  ~CodeGen_Spatial();

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

  std::string printFuncName(const Function *func, 
          std::map<Expr, std::string, ExprCompare> inputMap={}, 
          std::map<Expr, std::string, ExprCompare> outputMap={});

private:
  std::string restrictKeyword() const { return "restrict"; }
  std::string printDeclsAccel(std::map<Expr, std::string, ExprCompare> varMap,
                         std::vector<Expr> inputs, std::vector<Expr> outputs);

  std::string unpackTensorProperty(std::string varname, const GetProperty* op,
                              bool is_output_prop);
  std::string unpackTensorPropertyAccel(std::string varname, const GetProperty* op,
                              bool is_output_prop);

  std::string printInitMem(std::map<Expr, std::string, ExprCompare> varMap,
                         std::vector<Expr> inputs, std::vector<Expr> outputs);
  std::string outputInitMemArgs(std::string varname, const GetProperty* op,
                              bool is_output_prop, bool last);

  std::string printOutputCheck(std::map<std::tuple<Expr, TensorProperty, int, int>,
          std::string> outputProperties, std::vector<Expr> outputs);
  std::string outputCheckOutputArgs(std::string varname, Expr tnsr, TensorProperty property,
                            int mode, int index, bool last);

  // Used for printing out output store
  std::string printOutputStore(std::map<std::tuple<Expr, TensorProperty, int, int>,
          std::string> outputProperties, std::vector<Expr> outputs);
  std::string outputTensorProperty(std::string varname, Expr tnsr, TensorProperty property,
                            int mode, int index);
};
} // namespace ir
} // namespace taco
#endif
