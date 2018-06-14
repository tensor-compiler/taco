#ifndef TACO_ERROR_MESSAGES_H
#define TACO_ERROR_MESSAGES_H

#include <string>

namespace taco {
namespace error {

// unsupported type bit width error
extern const std::string type_mismatch;
extern const std::string type_bitwidt;

// TensorVar::setIndexExpression error messages
extern const std::string expr_dimension_mismatch;
extern const std::string expr_transposition;
extern const std::string expr_distribution;
extern const std::string expr_einsum_missformed;

// compile error messages
extern const std::string compile_without_expr;

// assemble error messages
extern const std::string assemble_without_compile;

// compute error messages
extern const std::string compute_without_compile;

// factory function error messages
extern const std::string requires_matrix;

#define INIT_REASON(reason) \
string reason_;             \
do {                        \
  if (reason == nullptr) {  \
    reason = &reason_;      \
  }                         \
  *reason = "";             \
} while (0)

}}
#endif
