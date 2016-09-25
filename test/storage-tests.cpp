#include "test.h"

#include "storage.h"


TEST(storage, vector) {
  TensorStorage vector;

  vector = TensorStorage("d");
  std::cout << vector << std::endl;
}

TEST(storage, matrix) {
  TensorStorage matrix;

  matrix = TensorStorage("dd");
  std::cout << matrix << std::endl;
}
