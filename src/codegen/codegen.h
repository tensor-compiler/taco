#ifndef TACO_CODEGEN_H
#define TACO_CODEGEN_H

#include <memory>
#include "taco/ir/ir.h"
#include "taco/ir/ir_printer.h"

namespace taco {
namespace ir {


class CodeGen : public IRPrinter {
public:
  /// Kind of output: header or implementation
  enum OutputKind { C99Header, C99Implementation };

  CodeGen(std::ostream& stream) : IRPrinter(stream) {};
  CodeGen(std::ostream& stream, bool color, bool simplify) : IRPrinter(stream, color, simplify) {};
  /// Initialize the default code generator
  static std::shared_ptr<CodeGen> init_default(std::ostream &dest, OutputKind outputKind);

  /// Compile a lowered function
  virtual void compile(Stmt stmt, bool isFirst=false) =0;
};

} // namespace ir
} // namespace taco
#endif
