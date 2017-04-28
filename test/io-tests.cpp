#include "test.h"

#include "taco/tensor.h"

using namespace taco;

TEST(io, tns) {
  TensorBase tensor = readTensor(testDataDirectory()+"3tensor.tns", "Foo");
  ASSERT_EQ("Foo", tensor.getName());
  tensor.pack();

  TensorBase expected(ComponentType::Double, {1073,1,7});
  expected.insert({736,  1, 1}, 1.0);
  expected.insert({1073, 1, 6}, 1.1);
  expected.insert({881,  1, 7}, 1.0);
  expected.pack();

  ASSERT_TRUE(equals(expected, tensor));
}

TEST(io, mtx) {
  TensorBase tensor = readTensor(testDataDirectory()+"2tensor.mtx", "Foo");
  ASSERT_EQ("Foo", tensor.getName());
  tensor.pack();

  TensorBase expected(ComponentType::Double, {32,32});
  expected.insert({0, 0}, 101.0);
  expected.insert({1, 0}, 102.0);
  expected.insert({5, 2}, 307.1);
  expected.pack();

  ASSERT_TRUE(equals(expected, tensor));
}
