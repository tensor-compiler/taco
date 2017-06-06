#ifndef TACO_ERROR_MESSAGES_H
#define TACO_ERROR_MESSAGES_H

#include <string>

namespace taco {
namespace error {

// Compile error messages
extern const std::string compile_without_expr;
extern const std::string compile_transposition;

// Compute error messages
extern const std::string compute_without_compile;

}}
#endif
