#ifndef TACO_CODEGEN_LEGION_C_H
#define TACO_CODEGEN_LEGION_C_H

#include "codegen.h"
#include "codegen_c.h"

namespace taco {
namespace ir {

class CodegenLegionC : public CodeGen_C {
public:
  CodegenLegionC(std::ostream &dest, OutputKind outputKind, bool simplify=true);

  void compile(Stmt stmt, bool isFirst=false) override;

private:
  std::string unpackTensorProperty(std::string varname, const GetProperty* op, bool is_output_prop) override;
  std::string printFuncName(const Function *func,
                            std::map<Expr, std::string, ExprCompare> inputMap,
                            std::map<Expr, std::string, ExprCompare> outputMap) override;

  void visit(const For* node) override;
  void visit(const Function* node) override;

  std::string taskArgsName(std::string taskName) {
    return taskName + "Args";
  }

  class FindVars;

  std::vector<Stmt> functions;
  std::vector<Expr> regionArgs;

  // Maps from tasks to packed arguments.
  std::map<Stmt, std::vector<Expr>> taskArgs;
};

}
}

#endif //TACO_CODEGEN_LEGION_C_H
