#ifndef TACO_EXPR_FACTORY_H
#define TACO_EXPR_FACTORY_H

#include <vector>

#include "taco/tensor.h"
#include "taco/format.h"

namespace taco {
namespace test {

struct ExprFactory {
  virtual Tensor<double> operator()(std::vector<Tensor<double>>&, Format) = 0;
  virtual ~ExprFactory() {};
};

struct VectorElwiseSqrtFactory : public ExprFactory {
  virtual Tensor<double> operator()(std::vector<Tensor<double>>&, Format);
  virtual ~VectorElwiseSqrtFactory() {};
};

struct MatrixElwiseMultiplyFactory : public ExprFactory {
  virtual Tensor<double> operator()(std::vector<Tensor<double>>&, Format);
  virtual ~MatrixElwiseMultiplyFactory() {};
};

struct MatrixMultiplyFactory : public ExprFactory {
  virtual Tensor<double> operator()(std::vector<Tensor<double>>&, Format);
  virtual ~MatrixMultiplyFactory() {};
};

struct MatrixTransposeMultiplyFactory : public ExprFactory {
  virtual Tensor<double> operator()(std::vector<Tensor<double>>&, Format);
  virtual ~MatrixTransposeMultiplyFactory() {};
};

struct MatrixColumnSquaredNormFactory : public ExprFactory {
  virtual Tensor<double> operator()(std::vector<Tensor<double>>&, Format);
  virtual ~MatrixColumnSquaredNormFactory() {};
};

struct MatrixColumnNormalizeFactory : public ExprFactory {
  virtual Tensor<double> operator()(std::vector<Tensor<double>>&, Format);
  virtual ~MatrixColumnNormalizeFactory() {};
};

struct MTTKRP1Factory : public ExprFactory {
  virtual Tensor<double> operator()(std::vector<Tensor<double>>&, Format);
  virtual ~MTTKRP1Factory() {};
};

struct MTTKRP2Factory : public ExprFactory {
  virtual Tensor<double> operator()(std::vector<Tensor<double>>&, Format);
  virtual ~MTTKRP2Factory() {};
};

struct MTTKRP3Factory : public ExprFactory {
  virtual Tensor<double> operator()(std::vector<Tensor<double>>&, Format);
  virtual ~MTTKRP3Factory() {};
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

struct KroneckerFactory : public ExprFactory {
  virtual Tensor<double> operator()(std::vector<Tensor<double>>&, Format);
  virtual ~KroneckerFactory() {};
};

}}
#endif
