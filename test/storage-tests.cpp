#include "test.h"

#include <iostream>
#include <map>

#include "tensor.h"
#include "format.h"
#include "util.h"

using namespace std;

TEST(storage, d) {
  Format format("d");

  Tensor<double, 1> vec1(format);
  ASSERT_EQ(1u, vec1.getOrder());
  vec1.insert({0}, 1.0);
  

  std::cout << vec1 << std::endl;

//  Tensor<double> vec5(format, 5);
//  for (size_t i=0; i < 5; ++i) {
//    ASSERT_FLOAT_EQ(0.0, vec5(i));
//  }
//  vec5(1) = 1.0;
//  vec5(4) = 2.0;
//  vec5.pack();
//  ASSERT_FLOAT_EQ(0.0, vec5(0));
//  ASSERT_FLOAT_EQ(1.0, vec5(1));
//  ASSERT_FLOAT_EQ(0.0, vec5(2));
//  ASSERT_FLOAT_EQ(0.0, vec5(3));
//  ASSERT_FLOAT_EQ(2.0, vec5(4));
}

TEST(storage, s) {
  Format format("s");

  std::map<int, double> vec1Vals;
  vec1Vals[0] = 1.0;

  Tensor<double, 1> vec1(format);
  ASSERT_EQ(1u, vec1.getOrder());
//  ASSERT_EQ(0, vec1.numNonZeroes());

  for (auto& val : vec1Vals) {
    vec1.insert({val.first}, val.second);
  }
  vec1.pack();
//  ASSERT_EQ(0, vec1.numNonZeroes());

//  for (auto& nonZero : vec1.nonZeroes()) {
//    ASSERT_FLOAT_EQ(vec1Vals[nonZero.getCoord()], vec1Vals[nonZero.getVal()]);
//  }


  std::map<int, double> vec5Vals;
  vec5Vals[1] = 1.0;
  vec5Vals[4] = 2.0;

//  Tensor<double> vec1(format, 5);
//  ASSERT_EQ(0, vec1.numNonZeroes());

//  for (auto& val : vec1Vals) {
//    std::cout << val.first << ": " << val.second << std::endl;
//    vec1(val.first) = val.second;
//  }
//  ASSERT_EQ(0, vec1.numNonZeroes());

//  for (auto& nonZero : vec1.nonZeroes()) {
//    ASSERT_FLOAT_EQ(vec1Vals[nonZero.getCoord()], vec1Vals[nonZero.getVal()]);
//  }
}
