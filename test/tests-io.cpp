#include "test.h"

#include "taco/tensor.h"

using namespace taco;

TEST(io, tns) {
  TensorBase tensor = read(testDataDirectory()+"3tensor.tns", Sparse);
  ASSERT_EQ(3, tensor.getOrder());
  for (ModeFormat modeType : tensor.getFormat().getModeFormats()) {
    ASSERT_EQ(Sparse, modeType);
  }

  TensorBase expected(Float64, {1073,1,7});
  expected.insert({735,  0, 0}, 1.0);
  expected.insert({1072, 0, 5}, 1.1);
  expected.insert({880,  0, 6}, 1.0);
  expected.pack();

  ASSERT_TRUE(equals(expected, tensor));
}

TEST(io, mtx) {
  TensorBase tensor = read(testDataDirectory()+"2tensor.mtx", Sparse);
  ASSERT_EQ(2, tensor.getOrder());
  for (ModeFormat modeType : tensor.getFormat().getModeFormats()) {
    ASSERT_EQ(Sparse, modeType);
  }

  TensorBase expected(Float64, {32,32});
  expected.insert({0, 0}, 101.0);
  expected.insert({1, 0}, 102.0);
  expected.insert({5, 2}, 307.1);
  expected.pack();

  ASSERT_TRUE(equals(expected, tensor));
}

TEST(io, tensor) {
  Tensor<double> tensor = read(testDataDirectory()+"3tensor.tns", Sparse);
  ASSERT_EQ(3, tensor.getOrder());
  for (ModeFormat modeType : tensor.getFormat().getModeFormats()) {
    ASSERT_EQ(Sparse, modeType);
  }

  TensorBase expected(Float64, {1073,1,7});
  expected.insert({735,  0, 0}, 1.0);
  expected.insert({1072, 0, 5}, 1.1);
  expected.insert({880,  0, 6}, 1.0);
  expected.pack();


  ASSERT_TRUE(equals(expected, tensor));
}

TEST(io, ttxdense) {
  Tensor<double> tensor = read(testDataDirectory()+"d432.ttx", Dense);
  ASSERT_EQ(3, tensor.getOrder());
  for (ModeFormat modeType : tensor.getFormat().getModeFormats()) {
    ASSERT_EQ(Dense, modeType);
  }

  TensorBase expected(Float64, {4,3,2}, Dense);
  expected.insert({0, 0, 0}, 1.0);
  expected.insert({1, 0, 0}, 2.0);
  expected.insert({2, 0, 0}, 3.0);
  expected.insert({3, 0, 0}, 4.0);
  expected.insert({0, 1, 0}, 5.0);
  expected.insert({1, 1, 0}, 6.0);
  expected.insert({2, 1, 0}, 7.0);
  expected.insert({3, 1, 0}, 8.0);
  expected.insert({0, 2, 0}, 9.0);
  expected.insert({1, 2, 0}, 10.0);
  expected.insert({2, 2, 0}, 11.0);
  expected.insert({3, 2, 0}, 12.0);
  expected.insert({0, 0, 1}, 13.0);
  expected.insert({1, 0, 1}, 14.0);
  expected.insert({2, 0, 1}, 15.0);
  expected.insert({3, 0, 1}, 16.0);
  expected.insert({0, 1, 1}, 17.0);
  expected.insert({1, 1, 1}, 18.0);
  expected.insert({2, 1, 1}, 19.0);
  expected.insert({3, 1, 1}, 20.0);
  expected.insert({0, 2, 1}, 21.0);
  expected.insert({1, 2, 1}, 22.0);
  expected.insert({2, 2, 1}, 23.0);
  expected.insert({3, 2, 1}, 24.0);
  expected.pack();


  ASSERT_TRUE(equals(expected, tensor));
}

TEST(io, ttxsparse) {
  Tensor<double> tensor = read(testDataDirectory()+"d567.ttx", Sparse);
  ASSERT_EQ(3, tensor.getOrder());
  for (ModeFormat modeType : tensor.getFormat().getModeFormats()) {
    ASSERT_EQ(Sparse, modeType);
  }

  TensorBase expected(Float64, {5,6,7}, Sparse);
  expected.insert({0, 0, 0}, 1.0);
  expected.insert({1, 2, 0}, 2.0);
  expected.insert({4, 0, 6}, 3.0);
  expected.insert({2, 5, 2}, 4.0);
  expected.pack();


  ASSERT_TRUE(equals(expected, tensor));
}

TEST(io, mtxsymmetric) {
  Tensor<double> tensor = read(testDataDirectory()+"ds33.mtx", Sparse);
  ASSERT_EQ(2, tensor.getOrder());
  for (ModeFormat modeType : tensor.getFormat().getModeFormats()) {
    ASSERT_EQ(Sparse, modeType);
  }

  TensorBase expected(Float64, {3,3}, Sparse);
  expected.insert({0, 1}, 1.0);
  expected.insert({0, 2}, 3.0);
  expected.insert({1, 0}, 1.0);
  expected.insert({1, 1}, 2.0);
  expected.insert({2, 0}, 3.0);
  expected.pack();


  ASSERT_TRUE(equals(expected, tensor));
}
