#ifndef TACO_UTIL_FILL_H
#define TACO_UTIL_FILL_H

#include <string>
#include <random>
#include <map>
#include <cstdlib>
#include <ctime>

#include "taco/tensor.h"

namespace taco {
namespace util {

enum class FillMethod {
  Dense,
  Uniform,
  Random,
  Sparse,
  SlicingH,
  SlicingV,
  FEM,
  HyperSparse,
  Blocked
};

// Some parameters
const std::map<FillMethod,double> fillFactors = {
    {FillMethod::Dense, 1.0},
    {FillMethod::Uniform, 1.0},
    {FillMethod::Random, 1.0},
    {FillMethod::Sparse, 0.07},
    {FillMethod::HyperSparse, 0.005},
    {FillMethod::SlicingH, 0.01},
    {FillMethod::SlicingV, 0.01},
    {FillMethod::FEM, 0.0}
};
const double doubleLowerBound = -10e6;
const double doubleUpperBound =  10e6;
const int blockDimension=4;
const FillMethod blockFillMethod=FillMethod::FEM;

void fillTensor(TensorBase& tens, const FillMethod& fill, double fillValue=-1.0);
void fillVector(TensorBase& tens, const FillMethod& fill, double fillValue);
void fillMatrix(TensorBase& tens, const FillMethod& fill, double fillValue);
void fillTensor3(TensorBase& tens, const FillMethod& fill, double fillValue);

void fillTensor(TensorBase& tens, const FillMethod& fill, double fillValue/*=-1.0*/) {
  double filling;
  if (fillValue==-1)
    filling=fillFactors.at(fill);
  else
    filling=fillValue;
  switch (tens.getOrder()) {
    case 1: {
      fillVector(tens, fill, filling);
      break;
    }
    case 2: {
      fillMatrix(tens, fill, filling);
      break;
    }
    case 3: {
      fillTensor3(tens, fill, filling);
      break;
    }
    default:
      taco_uerror << "Impossible to fill tensor " << tens.getName() <<
        " of order " << tens.getOrder() << std::endl;
  }
}

void fillVector(TensorBase& tensor, const FillMethod& fill, double fillValue) {
  // Random values
  std::uniform_real_distribution<double> unif(doubleLowerBound,
                                              doubleUpperBound);
  std::default_random_engine re;

  int vectorSize = tensor.getDimension(0);
  auto index = tensor.getStorage().getIndex();
  switch (fill) {
    case FillMethod::Dense: {
      for (int i = 0; i < vectorSize; ++i) {
        tensor.insert({i}, (double)i);
      }
      break;
    }
    case FillMethod::Uniform: {
      for (int i = 0; i < vectorSize; ++i) {
        tensor.insert({i}, 1.0);
      }
      break;
    }
    case FillMethod::Random: {
      for (int i = 0; i < vectorSize; ++i) {
        tensor.insert({i}, unif(re));
      }
      break;
    }
    case FillMethod::Sparse:
    case FillMethod::HyperSparse: {
      re.seed(std::random_device{}());

      // Random positions
      std::vector<int> positions(vectorSize);
      for (int i=0; i<vectorSize; i++) {
        positions.push_back(i);
      }
      srand(static_cast<unsigned>(time(NULL)));
      std::random_shuffle(positions.begin(),positions.end());

      int toFill=fillValue*vectorSize;
      for (int i=0; i<toFill; i++) {
        tensor.insert({positions[i]}, unif(re));
      }
      break;
    }
    default: {
      taco_uerror << "FillMethod not available for vectors" << std::endl;
      break;
    }
  }
  tensor.pack();
}

void fillMatrix(TensorBase& tens, const FillMethod& fill, double fillValue) {
  // Random values
  std::uniform_real_distribution<double> unif(doubleLowerBound,
                                              doubleUpperBound);
  std::default_random_engine re;
  re.seed(std::random_device{}());
  std::vector<int> tensorSize=tens.getDimensions();
  // Random positions
  std::vector<int> pos(tensorSize[0]*tensorSize[1]);
  for (int i=0; i < static_cast<int>(pos.size());i++){
    pos[i]=i;
  }
  std::random_shuffle(pos.begin(),pos.end());
  std::vector<std::vector<int>> positions(tens.getOrder());
  for (int j=0; j<tens.getOrder(); j++) {
    positions.push_back(std::vector<int>(tensorSize[j]));
    for (int i=0; i<tensorSize[j]; i++)
      positions[j].push_back(i);
    srand(static_cast<unsigned>(time(NULL)));
    std::random_shuffle(positions[j].begin(),positions[j].end());
  }
  switch (fill) {
    case FillMethod::Dense: {
      for (int i=0; i<tensorSize[0]; i++) {
        for (int j=0; j<(fillValue*tensorSize[1]); j++) {
          tens.insert({i,j}, unif(re));
        }
      }
      break;
    }
    case FillMethod::Uniform: {
      for (int i=0; i<tensorSize[0]; i++) {
        for (int j=0; j<(fillValue*tensorSize[1]); j++) {
          tens.insert({i,j}, 1.0);
        }
      }
      break;
    }
    case FillMethod::Sparse:
    case FillMethod::HyperSparse: {
      for (int i=0; i<(fillValue*tensorSize[0]); i++) {
        for (int j=0; j<(fillValue*tensorSize[1]); j++) {
          tens.insert({positions[0][i],positions[1][j]}, unif(re));
        }
        std::random_shuffle(positions[1].begin(),positions[1].end());
      }
      break;
    }
    case FillMethod::Random: {
      for (int i=0; i<(fillValue*pos.size()); i++) {
        tens.insert({pos[i]%tensorSize[1],pos[i]/tensorSize[1]}, unif(re));
      }
      break;
    }
    case FillMethod::SlicingH: {
      for (int i=0; i<(fillValue*tensorSize[0]); i++) {
        for (int j=0; j<(fillFactors.at(FillMethod::Dense)*tensorSize[1]); j++){
          tens.insert({positions[0][i],positions[1][j]}, unif(re));
        }
      }
      break;
    }
    case FillMethod::SlicingV: {
      for (int j=0; j<(fillValue*tensorSize[0]); j++) {
        for (int i=0; i<(fillFactors.at(FillMethod::Dense)*tensorSize[1]); i++){
          tens.insert({positions[0][i],positions[1][j]}, unif(re));
        }
      }
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
      break;
    }
    case FillMethod::Blocked: {
      std::vector<int> dimensions;
      dimensions.push_back(tensorSize[0]/blockDimension);
      dimensions.push_back(tensorSize[1]/blockDimension);
      Tensor<double> BaseTensor(tens.getName(), dimensions,
                                tens.getFormat());
      fillMatrix(BaseTensor, blockFillMethod, fillValue);
      for (const auto& elem : BaseTensor) {
        int row = (int)elem.first[0]*blockDimension;
        int col = (int)elem.first[1]*blockDimension;
        double value = elem.second;
        for (int i=0; i<blockDimension; i++) {
          for (int j=0; j<blockDimension; j++) {
            tens.insert({row+i,col+j},value/(i+1));
          }
        }
      }
      break;
    }
    default: {
      taco_uerror << "FillMethod not available for matrices" << std::endl;
      break;
    }
  }
  tens.pack();
}

void fillTensor3(TensorBase& tens, const FillMethod& fill, double fillValue) {
  // Random values
  std::uniform_real_distribution<double> unif(doubleLowerBound,
                                              doubleUpperBound);
  std::default_random_engine re;
  re.seed(std::random_device{}());
  std::vector<int> tensorSize=tens.getDimensions();
  // Random positions
  std::vector<int> pos(tensorSize[0]*tensorSize[1]*tensorSize[2]);
  for (int i=0; i < static_cast<int>(pos.size()); i++){
    pos[i]=i;
  }
  std::random_shuffle(pos.begin(),pos.end());
  std::vector<std::vector<int>> positions(tens.getOrder());
  for (int j=0; j<tens.getOrder(); j++) {
    positions.push_back(std::vector<int>(tensorSize[j]));
    for (int i=0; i<tensorSize[0]; i++)
      positions[j].push_back(i);
    srand(static_cast<unsigned>(time(0)));
    std::random_shuffle(positions[j].begin(),positions[j].end());
  }
  switch (fill) {
    case FillMethod::Dense: {
      for (int i=0; i<tensorSize[0]; i++) {
        for (int j=0; j<tensorSize[1]; j++) {
          for (int k=0; k<fillValue*tensorSize[2]; k++) {
            tens.insert({i,j,k}, unif(re));
          }
        }
      }
      tens.pack();
      break;
    }
    case FillMethod::Uniform: {
      for (int i=0; i<tensorSize[0]; i++) {
        for (int j=0; j<tensorSize[1]; j++) {
          for (int k=0; k<fillValue*tensorSize[2]; k++) {
            tens.insert({i,j,k}, 1.0);
          }
        }
      }
      tens.pack();
      break;
    }
    case FillMethod::Sparse:
    case FillMethod::HyperSparse: {
      for (int i=0; i<(fillValue*tensorSize[0]); i++) {
        for (int j=0; j<(fillValue*tensorSize[1]); j++) {
          for (int k=0; k<(fillValue*tensorSize[2]); k++) {
            tens.insert({positions[0][i],positions[1][j],positions[2][k]}, unif(re));
          }
          std::random_shuffle(positions[2].begin(),positions[2].end());
        }
        std::random_shuffle(positions[1].begin(),positions[1].end());
      }
      tens.pack();
      break;
    }
    case FillMethod::Random: {
      for (int i=0; i<(fillValue*pos.size()); i++) {
        tens.insert({pos[i]%tensorSize[1],(pos[i]/tensorSize[1])%tensorSize[2],(pos[i]/tensorSize[1])/tensorSize[2]}, unif(re));
      }
      tens.pack();
      break;
    }

    default: {
      taco_uerror << "FillMethod not available for tensors of order 3"
                  << std::endl;
      break;
    }
  }
}

}}
#endif
