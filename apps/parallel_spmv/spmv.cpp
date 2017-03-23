#include <iostream>
#include "taco/tensor.h"

using namespace taco;
int myCompute(void**);


int main(int argc, char* argv[]) {

  // Create formats
  Format csr({Dense,Sparse});
  Format  dv({Dense});

  // Create tensors
  Tensor<double> A({2,3},   csr);
  Tensor<double> c({3},     dv);
  Tensor<double> b({2},     dv);

  A.insert({0,0}, 1.0);

  c.insert(0, 4.0);
  c.insert(1, 5.0);

  // Pack data as described by the formats
  A.pack();
  c.pack();

  Var i, j(Var::Sum);
  b(i) = A(i,j) * c(j);

  // Compile the expression
  b.compile();
  
  b.assemble();
//  b.compute();
  b.computeWithFunc(myCompute);

  std::cout << b << std::endl;
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
    for (int i3A0 = 0; i3A0 < 2; i3A0 += 1) {
      A01_ptr = ((0 * 2) + i3A0);
      A21_ptr = ((0 * 2) + i3A0);

      ti4 = 0;
      for (int A02_ptr = A0_L1_ptr[A01_ptr]; A02_ptr < A0_L1_ptr[(A01_ptr + 1)]; A02_ptr += 1) {
        i4A0 = A0_L1_idx[A02_ptr];
        A11_ptr = ((0 * 3) + i4A0);

        ti4 = (ti4 + (A0_vals[A02_ptr] * A1_vals[A11_ptr]));


      }
      A2_vals[A21_ptr] = ti4;


    }

  ((double**)A2)[1]  = A2_vals;
  printf("Done with custom compute function\n");
  return 0;
}
