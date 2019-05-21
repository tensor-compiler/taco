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

  B(0,0,0) = 1.0;
  B(1,2,0) = 2.0;
  B(1,2,1) = 3.0;
  c(0) = 4.0;
  c(1) = 5.0;

  std::cout << "B.needsPack = " << B.needsPack() << std::endl;
  std::cout << "c.needsPack = " << c.needsPack() << std::endl;

  IndexVar i, j, k;
  A(i,j) = B(i,j,k) * c(k);

  std::cout << "A(i,j) = B(i,j,k) * c(k);" << std::endl;

  std::cout << "B.needsPack = " << B.needsPack() << std::endl;
  std::cout << "c.needsPack = " << c.needsPack() << std::endl;

  std::cout << "A.needsCompile = " << A.needsCompile() << std::endl;
  std::cout << "A.needsCompute = " << A.needsCompute() << std::endl;

  std::cout << "A.getDependentTensors" << std::endl;

  for (auto t : A.getDependentTensors()) {
    std::cout << t.getName() << std::endl;
  }

  std::cout << "B.getDependentTensors" << std::endl;

  for (auto t : B.getDependentTensors()) {
    std::cout << t.getName() << std::endl;
  }

  std::cout << "c.getDependentTensors" << std::endl;

  for (auto t : c.getDependentTensors()) {
    std::cout << t.getName() << std::endl;
  }

  // Modify on operand of A
  c(0) = 1.0;
  std::cout << "c(0) = 1.0;" << std::endl;

  std::cout << "A.needsCompile = " << A.needsCompile() << std::endl;
  std::cout << "A.needsCompute = " << A.needsCompute() << std::endl;

  std::cout << "B.needsPack = " << B.needsPack() << std::endl;
  std::cout << "c.needsPack = " << c.needsPack() << std::endl;

  std::cout << "A.getDependentTensors" << std::endl;

  for (auto t : A.getDependentTensors()) {
    std::cout << t.getName() << std::endl;
  }

  std::cout << "B.getDependentTensors" << std::endl;

  for (auto t : B.getDependentTensors()) {
    std::cout << t.getName() << std::endl;
  }

  std::cout << "c.getDependentTensors" << std::endl;

  for (auto t : c.getDependentTensors()) {
    std::cout << t.getName() << std::endl;
  }

  std::cout << "print values" << std::endl;
  for (auto val = A.beginTyped<int>(); val != A.endTyped<int>(); ++val) {
    std::cout << val->second << std::endl;
  }

  std::cout << "A.getDependentTensors" << std::endl;

  for (auto t : A.getDependentTensors()) {
    std::cout << t.getName() << std::endl;
  }

  std::cout << "B.getDependentTensors" << std::endl;

  for (auto t : B.getDependentTensors()) {
    std::cout << t.getName() << std::endl;
  }

  std::cout << "c.getDependentTensors" << std::endl;

  for (auto t : c.getDependentTensors()) {
    std::cout << t.getName() << std::endl;
  }
}
