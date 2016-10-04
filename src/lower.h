#ifndef TACO_LOWER_H
#define TACO_LOWER_H

#include "expr.h"
#include "ir.h"

namespace taco {
namespace internal {

Stmt lower(taco::Expr expr);

}}

#endif
