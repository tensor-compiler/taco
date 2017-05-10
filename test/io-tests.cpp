#include "test.h"

#include "taco/tensor.h"

using namespace taco;

TEST(io, tns) {
  TensorBase tensor = read(testDataDirectory()+"3tensor.tns", Sparse);
  ASSERT_EQ(3u, tensor.getOrder());
  for (auto& levelType : tensor.getFormat().getLevels()) {
    ASSERT_EQ(LevelType::Sparse, levelType.getType());
  }

  TensorBase expected(ComponentType::Double, {1073,1,7});
  expected.insert({735,  0, 0}, 1.0);
  expected.insert({1072, 0, 5}, 1.1);
  expected.insert({880,  0, 6}, 1.0);
  expected.pack();

  ASSERT_TRUE(equals(expected, tensor));
}

TEST(io, mtx) {
  TensorBase tensor = read(testDataDirectory()+"2tensor.mtx", Sparse);
  ASSERT_EQ(2u, tensor.getOrder());
  for (auto& levelType : tensor.getFormat().getLevels()) {
    ASSERT_EQ(LevelType::Sparse, levelType.getType());
  }

  TensorBase expected(ComponentType::Double, {32,32});
  expected.insert({0, 0}, 101.0);
  expected.insert({1, 0}, 102.0);
  expected.insert({5, 2}, 307.1);
  expected.pack();

  ASSERT_TRUE(equals(expected, tensor));
}

TEST(io, tensor) {
  Tensor<double> tensor = read(testDataDirectory()+"3tensor.tns", Sparse);
  ASSERT_EQ(3u, tensor.getOrder());
  for (auto& levelType : tensor.getFormat().getLevels()) {
    ASSERT_EQ(LevelType::Sparse, levelType.getType());
  }

  TensorBase expected(ComponentType::Double, {1073,1,7});
  expected.insert({735,  0, 0}, 1.0);
  expected.insert({1072, 0, 5}, 1.1);
  expected.insert({880,  0, 6}, 1.0);
  expected.pack();


  ASSERT_TRUE(equals(expected, tensor));
}
