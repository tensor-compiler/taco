#ifndef TACO_ERROR_MESSAGES_H
#define TACO_ERROR_MESSAGES_H

#include <string>

namespace taco {
namespace error {

// setExpr error messages
extern const std::string expr_transposition;
extern const std::string expr_distribution;

// compile error messages
extern const std::string compile_without_expr;

// assemble error messages
extern const std::string assemble_without_compile;

// compute error messages
extern const std::string compute_without_compile;

}}
#endif
