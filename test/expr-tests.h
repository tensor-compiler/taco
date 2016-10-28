#ifndef TACO_EXPR_TESTS_H
#define TACO_EXPR_TESTS_H

#include <set>
#include <vector>
#include <utility>

#include "tensor.h"
#include "var.h"
#include "expr.h"
#include "packed_tensor.h"
#include "operator.h"

namespace taco {
namespace test {

struct ExpressionGenerator {
  virtual Expr operator()(std::vector<Tensor<double>>&) = 0;
};

}}
#endif
