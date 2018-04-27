#include "expr_factory.h"

#include "taco/format.h"
#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/index_notation_nodes.h"

namespace taco {
namespace test {

typedef std::vector<Tensor<double>> Tensors;

Tensor<double>
VectorElwiseSqrtFactory::operator()(Tensors& operands, Format outFormat) {
  taco_iassert(operands.size() == 1);

  Tensor<double> A(operands[0].getDimensions(), outFormat);

  IndexVar i("i");
  A(i) = new SqrtNode(operands[0](i));

  return A;
}

Tensor<double>
MatrixElwiseMultiplyFactory::operator()(Tensors& operands, Format outFormat) {
  taco_iassert(operands.size() == 2);

  Tensor<double> A(operands[0].getDimensions(), outFormat);

  IndexVar i("i"), j("j");
  A(i,j) = operands[0](i,j) * operands[1](i,j);

  return A;
}

Tensor<double>
MatrixMultiplyFactory::operator()(Tensors& operands, Format outFormat) { 
  taco_iassert(operands.size() == 2);

  Tensor<double> A({operands[0].getDimension(0),
                    operands[1].getDimension(1)}, outFormat);

  IndexVar i("i"), j("j"), k("k");
  A(i,j) = operands[0](i,k) * operands[1](k,j);

  return A;
}

Tensor<double>
MatrixTransposeMultiplyFactory::operator()(Tensors& operands, 
                                           Format   outFormat) {
  taco_iassert(operands.size() == 1);

  Tensor<double> A(operands[0].getDimensions(), outFormat);

  IndexVar i("i"), j("j"), k("k");
  A(i,j) = operands[0](k,i) * operands[0](k,j);

  return A;
}

Tensor<double>
MatrixColumnSquaredNormFactory::operator()(Tensors& operands, 
                                           Format   outFormat) {
  taco_iassert(operands.size() == 1);

  Tensor<double> A({operands[0].getDimension(1)}, outFormat);

  IndexVar i("i"), j("j");
  A(i) = operands[0](j,i) * operands[0](j,i);

  return A;
}

Tensor<double>
MatrixColumnNormalizeFactory::operator()(Tensors& operands, Format outFormat) {
  taco_iassert(operands.size() == 2);

  Tensor<double> A(operands[0].getDimensions(), outFormat);

  IndexVar i("i"), j("j");
  A(i,j) = operands[0](i,j) / operands[1](j);

  return A;
}

Tensor<double>
MTTKRP1Factory::operator()(Tensors& operands, Format outFormat) {
  taco_iassert(operands.size() == 3);

  Tensor<double> A({operands[0].getDimension(0),
                    operands[1].getDimension(1)}, outFormat);

  IndexVar i("i"), j("j"), k("k"), l("l");
  A(i,j) = operands[0](i,k,l) * operands[2](l,j) * operands[1](k,j);

  return A;
}

Tensor<double>
MTTKRP2Factory::operator()(Tensors& operands, Format outFormat) {
  taco_iassert(operands.size() == 3);

  Tensor<double> A({operands[0].getDimension(1),
                    operands[1].getDimension(1)}, outFormat);

  IndexVar i("i"), j("j"), k("k"), l("l");
  A(i,j) = operands[0](k,i,l) * operands[2](l,j) * operands[1](k,j);

  return A;
}

Tensor<double>
MTTKRP3Factory::operator()(Tensors& operands, Format outFormat) {
  taco_iassert(operands.size() == 3);

  Tensor<double> A({operands[0].getDimension(2),
                    operands[1].getDimension(1)}, outFormat);

  IndexVar i("i"), j("j"), k("k"), l("l");
  A(i,j) = operands[0](k,l,i) * operands[2](l,j) * operands[1](k,j);

  return A;
}

Tensor<double>
TensorSquaredNormFactory::operator()(Tensors& operands, Format outFormat) {
  taco_iassert(operands.size() == 1);

  Tensor<double> A({}, outFormat);

  IndexVar i("i"), j("j"), k("k");
  A() = operands[0](i,j,k) * operands[0](i,j,k);

  return A;
}

Tensor<double>
FactorizedTensorSquaredNormFactory::operator()(Tensors& operands, 
                                               Format   outFormat) {
  taco_iassert(operands.size() == 4);

  Tensor<double> A({}, outFormat);

  IndexVar i("i"), j("j");
  A() = operands[0](i) * operands[0](j) * operands[1](i,j) * 
        operands[2](i,j) * operands[3](i,j);

  return A;
}

Tensor<double>
FactorizedTensorInnerProductFactory::operator()(Tensors& operands, 
                                                Format   outFormat) {
  taco_iassert(operands.size() == 5);

  Tensor<double> A({}, outFormat);

  IndexVar i("i"), j("j"), k("k"), r("r");
  A() = operands[0](i,j,k) * operands[1](r) * operands[2](i,r) * 
        operands[3](j,r) * operands[4](k,r);

  return A;
}

Tensor<double>
KroneckerFactory::operator()(Tensors& operands, Format outFormat) {
  taco_iassert(operands.size() == 2);

  Tensor<double> A({operands[0].getDimension(0),
                    operands[0].getDimension(1),
                    operands[1].getDimension(0),
                    operands[1].getDimension(1)}, outFormat);

  IndexVar i("i"), j("j"), k("k"), l("l");
  A(i,j,k,l) = operands[0](i,j) * operands[1](k,l);

  return A;
}

}}
