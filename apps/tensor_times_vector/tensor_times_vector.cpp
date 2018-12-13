#include <iostream>
#include "taco.h"

using namespace taco;

int main(int argc, char* argv[]) {
  // Create formats
  Format csr({Dense,Sparse});
  Format csf({Sparse,Sparse,Sparse});
  Format  sv({Sparse});

  // Create tensors
  Tensor<double> A({2,3},   csr);
  Tensor<double> B({2,3,4}, csf);
  Tensor<double> c({4},     sv);

  // Insert data into B and c
  B.insert({0,0,0}, 1.0);
  B.insert({1,2,0}, 2.0);
  B.insert({1,2,1}, 3.0);
  c.insert({0}, 4.0);
  c.insert({1}, 5.0);

  // Pack data as described by the formats
  B.pack();
  c.pack();

  // Form a tensor-vector multiplication expression
  IndexVar i, j, k;
  A(i,j) = B(i,j,k) * c(k);

  // Compile the expression
  A.compile();

  // Assemble A's indices and numerically compute the result
  A.assemble();
  A.compute();

  std::cout << A << std::endl;
}
