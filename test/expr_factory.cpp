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
MatrixTransposeMultiplyFactory::operator()(Tensors& operands, 
                                           Format   outFormat) {
  iassert(operands.size() == 1);

  Tensor<double> A(operands[0].getDimensions(), outFormat);

  Var i("i"), j("j"), k("k", Var::Sum);
  A(i,j) = operands[0](k,i) * operands[0](k,j);

  return A;
}

Tensor<double>
MTTKRPFactory::operator()(Tensors& operands, Format outFormat) {
  iassert(operands.size() == 3);

  Tensor<double> A({operands[0].getDimensions()[0],
                    operands[1].getDimensions()[1]}, outFormat);

  Var i("i"), j("j"), k("k", Var::Sum), l("l", Var::Sum);
  A(i,j) = operands[0](i,k,l) * operands[2](l,j) * operands[1](k,j);

  return A;
}

Tensor<double>
TensorSquaredNormFactory::operator()(Tensors& operands, Format outFormat) {
  iassert(operands.size() == 1);

  Tensor<double> A({}, outFormat);

  Var i("i", Var::Sum), j("j", Var::Sum), k("k", Var::Sum);
  A() = operands[0](i,j,k) * operands[0](i,j,k);

  return A;
}

Tensor<double>
FactorizedTensorSquaredNormFactory::operator()(Tensors& operands, 
                                               Format   outFormat) {
  iassert(operands.size() == 4);

  Tensor<double> A({}, outFormat);

  Var i("i", Var::Sum), j("j", Var::Sum);
  A() = operands[0](i) * operands[0](j) * operands[1](i,j) * 
        operands[2](i,j) * operands[3](i,j);

  return A;
}

Tensor<double>
FactorizedTensorInnerProductFactory::operator()(Tensors& operands, 
                                                Format   outFormat) {
  iassert(operands.size() == 5);

  Tensor<double> A({}, outFormat);

  Var i("i", Var::Sum), j("j", Var::Sum), k("k", Var::Sum), r("r", Var::Sum);
  A() = operands[0](i,j,k) * operands[1](r) * operands[2](i,r) * 
        operands[3](j,r) * operands[4](k,r);

  return A;
}

}}
