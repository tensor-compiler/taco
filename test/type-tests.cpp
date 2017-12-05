#include "test.h"
#include "taco/type.h"

using namespace taco;
using namespace std;

template <typename T> class IntTest : public ::testing::Test {};
TYPED_TEST_CASE_P(IntTest);
TYPED_TEST_P(IntTest, types) {
  DataType t = type<TypeParam>();
  ASSERT_EQ(DataType::Int, t.getKind());
  ASSERT_TRUE(t.isInt());
  ASSERT_EQ(sizeof(TypeParam)*8, t.getNumBits());
  ASSERT_EQ(sizeof(TypeParam), t.getNumBytes());
}
REGISTER_TYPED_TEST_CASE_P(IntTest, types);
typedef ::testing::Types<char, short, int, long, long long> GenericInts;
INSTANTIATE_TYPED_TEST_CASE_P(Generic, IntTest, GenericInts);
typedef ::testing::Types<int8_t, int16_t, int32_t, int64_t> SpecificInts;
INSTANTIATE_TYPED_TEST_CASE_P(specific, IntTest, SpecificInts);

template <typename T> class UIntTest : public ::testing::Test {};
TYPED_TEST_CASE_P(UIntTest);
TYPED_TEST_P(UIntTest, types) {
  DataType t = type<TypeParam>();
  ASSERT_EQ(DataType::UInt, t.getKind());
  ASSERT_TRUE(t.isUInt());
  ASSERT_EQ(sizeof(TypeParam)*8, t.getNumBits());
  ASSERT_EQ(sizeof(TypeParam), t.getNumBytes());
}
REGISTER_TYPED_TEST_CASE_P(UIntTest, types);
typedef ::testing::Types<unsigned char, unsigned short, unsigned int,
                         unsigned long, unsigned long long> GenericUInts;
INSTANTIATE_TYPED_TEST_CASE_P(Generic, UIntTest, GenericUInts);
typedef ::testing::Types<uint8_t, uint16_t, uint32_t, uint64_t> SpecificFloat;
INSTANTIATE_TYPED_TEST_CASE_P(specific, UIntTest, SpecificFloat);

template <typename T> class FloatTest : public ::testing::Test {};
TYPED_TEST_CASE_P(FloatTest);
TYPED_TEST_P(FloatTest, types) {
  DataType t = type<TypeParam>();
  ASSERT_EQ(DataType::Float, t.getKind());
  ASSERT_TRUE(t.isFloat());
  ASSERT_EQ(sizeof(TypeParam)*8, t.getNumBits());
  ASSERT_EQ(sizeof(TypeParam), t.getNumBytes());
}
REGISTER_TYPED_TEST_CASE_P(FloatTest, types);
typedef ::testing::Types<float, double> GenericFloat;
INSTANTIATE_TYPED_TEST_CASE_P(Generic, FloatTest, GenericFloat);

TEST(types, equality) {
  DataType fp32(DataType::Float, 32);
  DataType fp32_2(DataType::Float, 32);
  DataType fp64(DataType::Float, 64);
  DataType int32(DataType::Int, 32);
  DataType int64(DataType::Int, 64);
  DataType uint32(DataType::UInt, 32);

  ASSERT_TRUE(fp32 == fp32);
  ASSERT_TRUE(fp32 == fp32_2);
  ASSERT_TRUE(!(fp32 == fp64));
  ASSERT_TRUE(fp32 != fp64);
  ASSERT_TRUE(fp32 != int32);
  ASSERT_TRUE(int32 != uint32);
  ASSERT_TRUE(int32 != uint32);
}
