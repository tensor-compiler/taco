#include "expr_factory.h"

#include "format.h"
#include "tensor.h"
#include "var.h"
#include "expr.h"
#include "operator.h"

namespace taco {
namespace test {

typedef std::vector<Tensor<double>> Tensors;

Tensor<double>
MatrixElwiseMultiplyFactory::operator()(Tensors& operands, Format outFormat) {
  iassert(operands.size() == 2);

  Tensor<double> A(operands[0].getDimensions(), outFormat);

  Var i("i"), j("j");
  A(i,j) = operands[0](i,j) * operands[1](i,j);

  return A;
}

Tensor<double>
TensorInnerProductFactory::operator()(Tensors& operands, Format outFormat) {
  iassert(operands.size() == 1);

  Tensor<double> A({}, outFormat);

  Var i("i", Var::Sum), j("j", Var::Sum), k("k", Var::Sum);
  A() = operands[0](i,j,k) * operands[0](i,j,k);

  return A;
}

}}
