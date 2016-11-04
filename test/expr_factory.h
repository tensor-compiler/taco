#ifndef TACO_EXPR_FACTORY_H
#define TACO_EXPR_FACTORY_H

#include <vector>

#include "format.h"
#include "tensor.h"

namespace taco {
namespace test {

struct ExprFactory {
  virtual Tensor<double> operator()(std::vector<Tensor<double>>&, Format) = 0;
  virtual ~ExprFactory() {};
};

struct MatrixElwiseMultiplyFactory : public ExprFactory {
  virtual Tensor<double> operator()(std::vector<Tensor<double>>&, Format);
  virtual ~MatrixElwiseMultiplyFactory() {};
};

struct MatrixTransposeMultiplyFactory : public ExprFactory {
  virtual Tensor<double> operator()(std::vector<Tensor<double>>&, Format);
};

struct MTTKRPFactory : public ExprFactory {
  virtual Tensor<double> operator()(std::vector<Tensor<double>>&, Format);
};

struct TensorSquaredNormFactory : public ExprFactory {
  virtual Tensor<double> operator()(std::vector<Tensor<double>>&, Format);
};

struct FactorizedTensorSquaredNormFactory : public ExprFactory {
  virtual Tensor<double> operator()(std::vector<Tensor<double>>&, Format);
};

struct FactorizedTensorInnerProductFactory : public ExprFactory {
  virtual Tensor<double> operator()(std::vector<Tensor<double>>&, Format);
  virtual ~TensorInnerProductFactory() {};
};

}}
#endif
