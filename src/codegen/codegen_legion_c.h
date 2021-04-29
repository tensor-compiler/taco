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

  void visit(const For* node) override;

  std::vector<Stmt> functions;
};

}
}

#endif //TACO_CODEGEN_LEGION_C_H
