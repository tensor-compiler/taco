#include "test.h"
#include "taco/type.h"

using namespace taco;

template <typename T> class ScalarTensorTest : public ::testing::Test {};
TYPED_TEST_CASE_P(ScalarTensorTest);
TYPED_TEST_P(ScalarTensorTest, types) {
  Tensor<TypeParam> a("A");
  DataType t = type<TypeParam>();
  ASSERT_EQ(t, a.getComponentType());
}
REGISTER_TYPED_TEST_CASE_P(ScalarTensorTest, types);

typedef ::testing::Types<int8_t, int16_t, int32_t, int64_t, long long, uint8_t, uint16_t, uint32_t, uint64_t, unsigned long long, float, double, std::complex<float>, std::complex<double>> AllTypes;
INSTANTIATE_TYPED_TEST_CASE_P(tensor_types, ScalarTensorTest, AllTypes);


template <typename T> class ScalarValueTensorTest : public ::testing::Test {};
TYPED_TEST_CASE_P(ScalarValueTensorTest);
TYPED_TEST_P(ScalarValueTensorTest, types) {
  Tensor<TypeParam> a((TypeParam) 4.2);
  DataType t = type<TypeParam>();
  ASSERT_EQ(t, a.getComponentType());
  ASSERT_EQ((TypeParam) 4.2, a.begin()->second);
}
REGISTER_TYPED_TEST_CASE_P(ScalarValueTensorTest, types);
INSTANTIATE_TYPED_TEST_CASE_P(tensor_types, ScalarValueTensorTest, AllTypes);


template <typename T> class VectorTensorTest : public ::testing::Test {};
TYPED_TEST_CASE_P(VectorTensorTest);
TYPED_TEST_P(VectorTensorTest, types) {
  Tensor<TypeParam> a({5}, Sparse);
  DataType t = type<TypeParam>();
  ASSERT_EQ(t, a.getComponentType());
  ASSERT_EQ(1u, a.getOrder());
  ASSERT_EQ(5,  a.getDimension(0));
  
  map<vector<int>,TypeParam> vals = {{{0}, 1.0}, {{2}, 2.0}};
  for (auto& val : vals) {
    a.insert(val.first, val.second);
  }
  a.pack();
  
  for (auto& val : a) {
    ASSERT_TRUE(util::contains(vals, val.first));
    ASSERT_EQ(vals.at(val.first), val.second);
  }
  
  TensorBase abase = a;
  for (auto& val : iterate<TypeParam>(abase)) {
    ASSERT_TRUE(util::contains(vals, val.first));
    ASSERT_EQ(vals.at(val.first), val.second);
  }
}
REGISTER_TYPED_TEST_CASE_P(VectorTensorTest, types);
INSTANTIATE_TYPED_TEST_CASE_P(tensor_types, VectorTensorTest, AllTypes);


template <typename T> class IterateTensorTest : public ::testing::Test {};
TYPED_TEST_CASE_P(IterateTensorTest);
TYPED_TEST_P(IterateTensorTest, types) {
  Tensor<TypeParam> a({5}, Sparse);
  a.insert({1}, 10.0);
  a.pack();
  ASSERT_TRUE(a.begin() != a.end());
  ASSERT_TRUE(++a.begin() == a.end());
  ASSERT_EQ((TypeParam) 10.0, (a.begin()++)->second);
}
REGISTER_TYPED_TEST_CASE_P(IterateTensorTest, types);
INSTANTIATE_TYPED_TEST_CASE_P(tensor_types, IterateTensorTest, AllTypes);
