#ifndef TACO_IR_VERIFIER_H
#define TACO_IR_VERIFIER_H

#include "taco/ir/ir.h"
#include "taco/ir/ir_visitor.h"

namespace taco {
namespace ir {
/** Verify an Stmt is well-formed for codegen */
bool verify(const Stmt s, std::string *message);

/** Verify an Stmt is well-formed for codegen */
bool verify(const Expr e, std::string *message);

} // namespace ir
} // namespace taco

#endif // TACO_IR_VERIFIER_H
