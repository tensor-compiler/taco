#include <iostream>
#include "taco.h"

using namespace taco;

int main(int argc, char* argv[]) {
  Format csr({Dense,Sparse});
  Format csf({Sparse,Sparse,Sparse});
  Format  sv({Sparse});

  Tensor<double> A("A", {2,3},   csr);
  Tensor<double> B("B", {2,3,4}, csf);
  Tensor<double> c("c", {4},     sv);

  // Insert data into B and c
  B(0,0,0) = 1.0;
  B(1,2,0) = 2.0;
  B(1,2,1) = 3.0;
  c(0) = 4.0;
  c(1) = 5.0;

  IndexVar i, j, k;
  A(i,j) = B(i,j,k) * c(k);

  std::cout << A << std::endl;
}
