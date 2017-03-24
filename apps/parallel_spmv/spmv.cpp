#include <iostream>
#include "taco/tensor.h"
#include <omp.h>
#include <random>

using namespace taco;
int myCompute(void**);
int myCompute2(void**);
bool compare(const Tensor<double>&tensor,
             const Tensor<double>&tensor2);


int main(int argc, char* argv[]) {
  // Create formats
  Format csr({Dense,Sparse});
  Format  dv({Dense});

  // Create tensors
  Tensor<double> A({50,25},   csr);
  Tensor<double> c({25},     dv);
  Tensor<double> b({50},     dv);
  Tensor<double> b2({50},     dv);

  for (int i=0; i<10; i++) {
    A.insert({rand() % 50, rand() % 25}, 1.0);
  }
  A.insert({0,0}, 1.0);

  for (int i=0; i<25; i++) {
    c.insert({i}, 4.0);
  }


  // Pack data as described by the formats
  A.pack();
  c.pack();

  Var i, j(Var::Sum);
  b(i) = A(i,j) * c(j);
  b2(i) = A(i,j) * c(j);
  

  // Compile the expression
  b.compile();
  b2.compile();
  
  b.assemble();
  b2.assemble();
//  b.compute();
  b.computeWithFunc(myCompute);
  
  std::cout << "A\n";
  std::cout << A << std::endl;
  
  std::cout << "c\n";
  std::cout << c << std::endl;

  
  std::cout << "b (parallel)\n";
  std::cout << b << std::endl;
  
  b2.compute();
  std::cout << compare(b, b2) << std::endl;
  
  return 0;
}


typedef std::set<typename Tensor<double>::Value> Values;
bool compare(const Tensor<double>&tensor,
             const Tensor<double>&tensor2)  {
  if (tensor.getDimensions() != tensor2.getDimensions()) {
    return false;
  }

  {
    std::set<typename Tensor<double>::Coordinate> coords;
    for (const auto& val : tensor) {
      if (!coords.insert(val.first).second) {
        return false;
      }
    }
  }

  Values vals;
  Values vals2;
  for (const auto& val : tensor) {
    if (val.second != 0) {
      vals.insert(val);
    }
  }
  for (const auto& val : tensor2) {
    if (val.second != 0) {
      vals2.insert(val);
    }
  }

  return vals == vals2;
}



#define restrict __restrict__
int myCompute(void** inputPack)   {
  void** A2 = &(inputPack[0]);
  void** A0 = &(inputPack[2]);
  void** A1 = &(inputPack[6]);
  int A01_ptr;
  int i3A0;
  int A02_ptr;
  int i4A0;
  int A11_ptr;
  int A21_ptr;
  double* restrict A2_vals = (double*)A2[1];
  double ti4;
  int* restrict A0_L1_idx = (int*)A0[2];
  double* restrict A0_vals = (double*)A0[3];
  double* restrict A1_vals = (double*)A1[1];
  int* restrict A0_L1_ptr = (int*)A0[1];
    /* A2(i3) = (A0(i3, i4) * A1(i4)) */
    #pragma omp parallel for private(A01_ptr,A21_ptr,ti4,i4A0,A11_ptr)
    //pragma omp parallel for
    for (int i3A0 = 0; i3A0 < 50; i3A0 += 1) {
      A01_ptr = ((0 * 50) + i3A0);
      A21_ptr = ((0 * 50) + i3A0);

      ti4 = 0;
      for (int A02_ptr = A0_L1_ptr[A01_ptr]; A02_ptr < A0_L1_ptr[(A01_ptr + 1)]; A02_ptr += 1) {
        i4A0 = A0_L1_idx[A02_ptr];
        A11_ptr = ((0 * 25) + i4A0);

        ti4 = (ti4 + (A0_vals[A02_ptr] * A1_vals[A11_ptr]));


      }
      A2_vals[A21_ptr] = ti4;


    }

  ((double**)A2)[1]  = A2_vals;
  std::cout << "[Paralel] Done\n";
  return 0;
}

#define restrict __restrict__
int myCompute2(void** inputPack)   {
  void** A2 = &(inputPack[0]);
  void** A0 = &(inputPack[2]);
  void** A1 = &(inputPack[6]);
  int A01_ptr;
  int i3A0;
  int A02_ptr;
  int i4A0;
  int A11_ptr;
  int A21_ptr;
  double* restrict A2_vals = (double*)A2[1];
  double ti4;
  int* restrict A0_L1_idx = (int*)A0[2];
  double* restrict A0_vals = (double*)A0[3];
  double* restrict A1_vals = (double*)A1[1];
  int* restrict A0_L1_ptr = (int*)A0[1];
    /* A2(i3) = (A0(i3, i4) * A1(i4)) */
    for (int i3A0 = 0; i3A0 < 50; i3A0 += 1) {
      A01_ptr = ((0 * 50) + i3A0);
      A21_ptr = ((0 * 50) + i3A0);

      ti4 = 0;
      for (int A02_ptr = A0_L1_ptr[A01_ptr]; A02_ptr < A0_L1_ptr[(A01_ptr + 1)]; A02_ptr += 1) {
        i4A0 = A0_L1_idx[A02_ptr];
        A11_ptr = ((0 * 25) + i4A0);

        ti4 = (ti4 + (A0_vals[A02_ptr] * A1_vals[A11_ptr]));


      }
      A2_vals[A21_ptr] = ti4;


    }

  ((double**)A2)[1]  = A2_vals;
    std::cout << "[Serial] Done\n";

  return 0;
}

