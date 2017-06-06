#ifndef TACO_ERROR_CHECKS_H
#define TACO_ERROR_CHECKS_H

#include <vector>

namespace taco {
class TensorBase;

namespace error {

/// Returns true iff the tensors's index expression contains a transposition.
bool containsTranspose(const TensorBase& tensor);

}}
#endif
