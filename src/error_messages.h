#ifndef TACO_ERROR_MESSAGES_H
#define TACO_ERROR_MESSAGES_H

#include <string>

namespace taco {
namespace error {

const std::string compute_without_compile =
    "The compile method must be called before compute.";

}}
#endif
