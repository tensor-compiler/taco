#include "test.h"

#include <iostream>
#include <map>

#include "tensor.h"
#include "format.h"
#include "util.h"

#include "packed_tensor.h"

using namespace std;

TEST(storage, d5) {
  Format format("d");

  Tensor<double, 1> vec1(format);
  ASSERT_EQ(1u, vec1.getOrder());
  vec1.insert({0}, 1.0);
  vec1.pack();
  ASSERT_EQ(1u, vec1.getPackedTensor()->getNnz());

  Tensor<double, 5> vec5(format);
  vec5.pack();
  ASSERT_EQ(5u, vec5.getPackedTensor()->getNnz());
  vec5.insert({4}, 2.0);
  vec5.insert({1}, 1.0);
  std::cout << vec5 << std::endl;
  vec5.pack();
  ASSERT_EQ(5u, vec5.getPackedTensor()->getNnz());

  std::cout << vec5 << std::endl;
}

TEST(storage, s) {
  Format format("s");

  Tensor<double, 1> vec1(format);
  ASSERT_EQ(1u, vec1.getOrder());
  vec1.insert({0}, 1.0);
  vec1.pack();
  ASSERT_EQ(1u, vec1.getPackedTensor()->getNnz());

  Tensor<double, 5> vec5(format);
  vec5.pack();
  ASSERT_EQ(0u, vec5.getPackedTensor()->getNnz());
  vec5.insert({4}, 2.0);
  vec5.insert({1}, 1.0);
  std::cout << vec5 << std::endl;
  vec5.pack();
  ASSERT_EQ(2u, vec5.getPackedTensor()->getNnz());

  std::cout << vec5 << std::endl;
}
