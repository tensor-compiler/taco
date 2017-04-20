#include "test.h"

#include "taco/tensor.h"

using namespace taco;

TEST(io, tns_3tensor) {
  TensorBase tensor = readTensor(testDataDirectory() + "3tensor.tns", "Foo");
  ASSERT_EQ("Foo", tensor.getName());
  tensor.pack();

  TensorBase expected(ComponentType::Double, {1073,1,7});
  expected.insert({736, 1, 1},  1.0);
  expected.insert({1073, 1, 6}, 1.0);
  expected.insert({881, 1, 7,}, 1.0);
  expected.pack();

  ASSERT_TRUE(equals(expected, tensor));
}
