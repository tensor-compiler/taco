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
  virtual ~MatrixTransposeMultiplyFactory() {};
};

struct MTTKRPFactory : public ExprFactory {
  virtual Tensor<double> operator()(std::vector<Tensor<double>>&, Format);
  virtual ~MTTKRPFactory() {};
};

struct TensorSquaredNormFactory : public ExprFactory {
  virtual Tensor<double> operator()(std::vector<Tensor<double>>&, Format);
  virtual ~TensorSquaredNormFactory() {};
};

struct FactorizedTensorSquaredNormFactory : public ExprFactory {
  virtual Tensor<double> operator()(std::vector<Tensor<double>>&, Format);
  virtual ~FactorizedTensorSquaredNormFactory() {};
};

struct FactorizedTensorInnerProductFactory : public ExprFactory {
  virtual Tensor<double> operator()(std::vector<Tensor<double>>&, Format);
  virtual ~FactorizedTensorInnerProductFactory() {};
};

}}
#endif
