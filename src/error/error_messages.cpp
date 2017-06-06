#include "error_messages.h"

namespace taco {
namespace error {

const std::string compile_without_expr =
    "An index expression must be defined before compile is called.";

const std::string compile_transposition =
    "Transpositions in computations are not currently supported, but are "
    "planned for the future.";

const std::string compute_without_compile =
    "The compile method must be called before compute.";

}}
