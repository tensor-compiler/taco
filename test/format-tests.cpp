#include "test.h"

#include "format.h"

TEST(storage, vector) {
  Format vector;

  vector = Format("d");
  std::cout << vector << std::endl;
}

TEST(storage, matrix) {
  Format matrix;

  matrix = Format("dd");
  std::cout << matrix << std::endl;
}
