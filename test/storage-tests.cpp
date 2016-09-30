#include "test.h"

#include <iostream>
#include <map>

#include "tensor.h"
#include "format.h"
#include "util.h"

#include "packed_tensor.h"

using namespace std;

template <typename T>
void ASSERT_ARRAY_EQ(const T* actual, vector<T> expected) {
  for (size_t i=0; i < expected.size(); ++i) {
    ASSERT_FLOAT_EQ(expected[i], ((T*)actual)[i]);
  }
}

TEST(DISABLED_storage, d5) {
  Format format("d");

//  Tensor<double, 1> vec1(format);
//  ASSERT_EQ(1u, vec1.getOrder());
//  vec1.insert({0}, 1.0);
//  vec1.pack();
//  ASSERT_EQ(1u, vec1.getPackedTensor()->getNnz());

  Tensor<double, 5> vec5(format);
//  vec5.pack();
//  ASSERT_EQ(5u, vec5.getPackedTensor()->getNnz());
  vec5.insert({4}, 2.0);
  vec5.insert({1}, 1.0);
  std::cout << vec5 << std::endl << std::endl;
  vec5.pack();

  auto vec5p = vec5.getPackedTensor();
  ASSERT_EQ(5u, vec5p->getNnz());
  ASSERT_ARRAY_EQ((double*)vec5p->getValues(), {0.0, 1.0, 0.0, 0.0, 4.0});

  auto indices = vec5p->getIndices();
  ASSERT_EQ(0u, indices.size());
}

TEST(storage, s) {
  Format format("s");

//  Tensor<double, 1> vec1(format);
//  ASSERT_EQ(1u, vec1.getOrder());
//  vec1.insert({0}, 1.0);
//  vec1.pack();
//  ASSERT_EQ(1u, vec1.getPackedTensor()->getNnz());

  Tensor<double, 5> vec5(format);
  ASSERT_EQ(1u, vec5.getOrder());
  ASSERT_EQ(nullptr, vec5.getPackedTensor());
//  vec5.pack();
//  ASSERT_EQ(0u, vec5.getPackedTensor()->getNnz());
  vec5.insert({4}, 2.0);
  vec5.insert({1}, 1.0);
  std::cout << vec5 << std::endl << std::endl;
  vec5.pack();

  auto vec5p = vec5.getPackedTensor();
  ASSERT_EQ(2u, vec5p->getNnz());
  ASSERT_ARRAY_EQ((double*)vec5p->getValues(), {1.0, 4.0});

  auto indices = vec5p->getIndices();
  ASSERT_EQ(1u, indices.size());
  auto indexArrays = indices[0];
  ASSERT_EQ(2u, indexArrays.size());
  ASSERT_ARRAY_EQ(indexArrays[0], {1, 4});
}
