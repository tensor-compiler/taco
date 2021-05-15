#ifndef TACO_CODEGEN_LEGION_C_H
#define TACO_CODEGEN_LEGION_C_H

#include "codegen.h"
#include "codegen_c.h"
#include "codegen_legion.h"

namespace taco {
namespace ir {

class CodegenLegionC : public CodeGen_C, public CodegenLegion {
public:
  CodegenLegionC(std::ostream &dest, OutputKind outputKind, bool simplify=true);
  void compile(Stmt stmt, bool isFirst=false) override;

private:
  // TODO (rohany): It doesn't seem like I can override these.
  using IRPrinter::visit;
  void visit(const For* node) override;
  void visit(const Function* node) override;
  void visit(const PackTaskArgs* node) override;
  void emitHeaders(std::ostream& o) override;
  // TODO (rohany): It also doesn't seem like I can avoid duplicating this class.
  class FindVars;
  Stmt stmt;
};

}
}

#endif //TACO_CODEGEN_LEGION_C_H
