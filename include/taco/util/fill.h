#ifndef TACO_UTIL_FILL_H
#define TACO_UTIL_FILL_H

#include <string>
#include <random>

#include "taco/tensor.h"

namespace taco {
namespace util {

enum class FillMethod {
  Dense,
  Sparse,
  SlicingH,
  SlicingV,
  FEM,
  HyperSpace,
  Blocked
};

// Some parameters
const std::map<FillMethod,double> fillFactors = {
    {FillMethod::Dense, 1.0},
    {FillMethod::Sparse, 0.07},
    {FillMethod::HyperSpace, 0.005},
    {FillMethod::SlicingH, 0.01},
    {FillMethod::SlicingV, 0.01}
};
const double doubleLowerBound = -10e6;
const double doubleUpperBound =  10e6;
const int blockDim=4;
const FillMethod blockFillMethod=FillMethod::FEM;

void fillTensor(TensorBase& tens, const FillMethod& fill);
void fillVector(TensorBase& tens, const FillMethod& fill);
void fillMatrix(TensorBase& tens, const FillMethod& fill);

void fillTensor(TensorBase& tens, const FillMethod& fill) {
  switch (tens.getOrder()) {
    case 1: {
      fillVector(tens, fill);
      break;
    }
    case 2: {
      fillMatrix(tens, fill);
      break;
    }
    default:
      taco_uerror << "Impossible to fill tensor " << tens.getName() <<
        " of dimension " << tens.getOrder() << std::endl;
  }
}

void fillVector(TensorBase& tensor, const FillMethod& fill) {
  // Random values
  std::uniform_real_distribution<double> unif(doubleLowerBound,
                                              doubleUpperBound);
  std::default_random_engine re;
  re.seed(std::random_device{}());
  int vectorSize=tensor.getDimensions()[0];
  // Random positions
  std::vector<int> positions(vectorSize);
  for (int i=0; i<vectorSize; i++)
    positions.push_back(i);
  srand(time(0));
  std::random_shuffle(positions.begin(),positions.end());
  switch (fill) {
    case FillMethod::Dense: {
      auto num = tensor.getStorage().getSize().values;
      tensor.getStorage().setValues((double*)malloc(num * sizeof(double)));
      double* values = (double*)tensor.getStorage().getValues();
      for (size_t i=0; i<num; i++) {
        values[i] = unif(re);
      }
      break;
    }
    case FillMethod::Sparse:
    case FillMethod::HyperSpace: {
      int toFill=fillFactors.at(fill)*vectorSize;
      for (int i=0; i<toFill; i++) {
        tensor.insert({positions[i]}, unif(re));
      }
      tensor.pack();
      break;
    }
    default: {
      taco_uerror << "FillMethod not available for vectors" << std::endl;
      break;
    }
  }
}

void fillMatrix(TensorBase& tens, const FillMethod& fill) {
  // Random values
  std::uniform_real_distribution<double> unif(doubleLowerBound,
                                              doubleUpperBound);
  std::default_random_engine re;
  re.seed(std::random_device{}());
  std::vector<int> tensorSize=tens.getDimensions();
  // Random positions
  std::vector<std::vector<int>> positions(tens.getOrder());
  for (size_t j=0; j<tens.getOrder(); j++) {
    positions.push_back(std::vector<int>(tensorSize[j]));
    for (int i=0; i<tensorSize[0]; i++)
      positions[j].push_back(i);
    srand(time(0));
    std::random_shuffle(positions[j].begin(),positions[j].end());
  }
  switch (fill) {
    case FillMethod::Dense: {
      for (int i=0; i<tensorSize[0]; i++) {
        for (int j=0; j<(fillFactors.at(fill)*tensorSize[1]); j++) {
          tens.insert({i,positions[1][j]}, unif(re));
        }
        std::random_shuffle(positions[1].begin(),positions[1].end());
      }
      tens.pack();
      break;
    }
    case FillMethod::Sparse:
    case FillMethod::HyperSpace: {
      for (int i=0; i<(fillFactors.at(fill)*tensorSize[0]); i++) {
        for (int j=0; j<(fillFactors.at(fill)*tensorSize[1]); j++) {
          tens.insert({positions[0][i],positions[1][j]}, unif(re));
        }
        std::random_shuffle(positions[1].begin(),positions[1].end());
      }
      tens.pack();
      break;
    }
    case FillMethod::SlicingH: {
      for (int i=0; i<(fillFactors.at(fill)*tensorSize[0]); i++) {
        for (int j=0; j<(fillFactors.at(FillMethod::Dense)*tensorSize[1]); j++){
          tens.insert({positions[0][i],positions[1][j]}, unif(re));
        }
      }
      tens.pack();
      break;
    }
    case FillMethod::SlicingV: {
      for (int j=0; j<(fillFactors.at(fill)*tensorSize[0]); j++) {
        for (int i=0; i<(fillFactors.at(FillMethod::Dense)*tensorSize[1]); i++){
          tens.insert({positions[0][i],positions[1][j]}, unif(re));
        }
      }
      tens.pack();
      break;
    }
    case FillMethod::FEM: {
      for (int i=0; i<tensorSize[0]-1; i++) {
        tens.insert({i,i}, unif(re));
        double value = unif(re);
        tens.insert({i+1,i}, value);
        tens.insert({i,i+1}, value);
        if (i<tensorSize[0]-3) {
          value = unif(re);
          tens.insert({i+3,i}, value);
          tens.insert({i,i+3}, value);
        }
      }
      tens.insert({tensorSize[0]-1,tensorSize[0]-1}, unif(re));
      tens.pack();
      break;
    }
    case FillMethod::Blocked: {
      vector<int> dimensionSizes;
      dimensionSizes.push_back(tensorSize[0]/blockDim);
      dimensionSizes.push_back(tensorSize[1]/blockDim);
      Tensor<double> BaseTensor(tens.getName(), dimensionSizes,
                                tens.getFormat(), DEFAULT_ALLOC_SIZE);
      fillMatrix(BaseTensor, blockFillMethod);
      for (const auto& elem : BaseTensor) {
        int row = elem.first[0]*blockDim;
        int col = elem.first[1]*blockDim;
        double value = elem.second;
        for (int i=0; i<blockDim; i++) {
          for (int j=0; j<blockDim; j++) {
            tens.insert({row+i,col+j},value/(i+1));
          }
        }
      }
      tens.pack();
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
