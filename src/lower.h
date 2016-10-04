#ifndef TACIT_LOWER_H
#define TACIT_LOWER_H

#include "expr.h"
#include "ir.h"

namespace tacit {
namespace internal {

Stmt lower(tacit::Expr expr);

}}

#endif
