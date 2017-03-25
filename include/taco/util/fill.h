#ifndef TACO_UTIL_FILL_H
#define TACO_UTIL_FILL_H

#include <string>
#include <random>

namespace taco {
namespace util {

enum class FillMethod {
  Dense,
  Sparse,
  Slicing,
  FEM,
  HyperSpace
};

const double doubleLowerBound = -10e6;
const double doubleUpperBound =  10e6;

void fillTensor(TensorBase& tens, const FillMethod& fill, int dim);
void fillVector(TensorBase& tens, const FillMethod& fill, int dim);
void fillMatrix(TensorBase& tens, const FillMethod& fill, int dim);

void fillTensor(TensorBase& tens, const FillMethod& fill, int dim) {
  switch (tens.getOrder()) {
    case 1: {
      fillVector(tens, fill, dim);
      break;
    }
    case 2: {
      fillMatrix(tens, fill, dim);
      break;
    }
    default:
      taco_uerror << "Impossible to fill tensor " << tens.getName() <<
        " of dimension " << tens.getOrder() << std::endl;
  }
}

void fillVector(TensorBase& tens, const FillMethod& fill, int dim) {
  switch (fill) {
    case FillMethod::Dense: {
      std::uniform_real_distribution<double> unif(doubleLowerBound,
                                                  doubleUpperBound);
      std::default_random_engine re;
      re.seed(std::random_device{}());
      int toFill=std::min(dim,int(tens.getStorage().getSize().values));
      for (int i=0; i<toFill; i++) {
        tens.insert({i}, unif(re));
      }
      tens.pack();
      break;
    }
    case FillMethod::Sparse:
    case FillMethod::HyperSpace: {
      taco_not_supported_yet;
      break;
    }
    default: {
      taco_uerror << "FillMethod not available for vectors" << std::endl;
      break;
    }
  }
}

void fillMatrix(TensorBase& tens, const FillMethod& fill, int dim) {
  switch (fill) {
    case FillMethod::Dense: {
      std::uniform_real_distribution<double> unif(doubleLowerBound,
                                                  doubleUpperBound);
      std::default_random_engine re;
      re.seed(std::random_device{}());
      int toFill=std::min(dim,int(tens.getStorage().getSize().values));
      for (int i=0; i<toFill; i++) {
        for (int j=0; j<toFill; j++) {
          tens.insert({i,j}, unif(re));
        }
      }
      tens.pack();
      break;
    }
    case FillMethod::Slicing:
    case FillMethod::FEM:
    case FillMethod::HyperSpace: {
      taco_not_supported_yet;
      break;
    }
    default: {
      taco_uerror << "FillMethod not available for matrices" << std::endl;
      break;
    }
  }

}

}}
#endif
