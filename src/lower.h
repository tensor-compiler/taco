#ifndef TACO_LOWER_H
#define TACO_LOWER_H

#include "expr.h"
#include "ir.h"

namespace taco {
namespace internal {
class IterationSchedule;

Stmt lower(std::string name, std::vector<taco::Var> indexVars, taco::Expr expr,
           const IterationSchedule& schedule);

}}

#endif
