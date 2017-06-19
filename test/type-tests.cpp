#include "test.h"
#include "taco/type.h"

using namespace taco;
using namespace std;

template <typename T> class Int : public ::testing::Test {};
TYPED_TEST_CASE_P(Int);
TYPED_TEST_P(Int, types) {
  Type t = typeOf<TypeParam>();
  ASSERT_EQ(Type::Int, t.getKind());
  ASSERT_TRUE(t.isInt());
  ASSERT_EQ(sizeof(TypeParam)*8, t.getNumBits());
  ASSERT_EQ(sizeof(TypeParam), t.getNumBytes());
}
REGISTER_TYPED_TEST_CASE_P(Int, types);
typedef ::testing::Types<char, short, int, long, long long> GenericInts;
INSTANTIATE_TYPED_TEST_CASE_P(Generic, Int, GenericInts);
typedef ::testing::Types<int8_t, int16_t, int32_t, int64_t> SpecificInts;
INSTANTIATE_TYPED_TEST_CASE_P(specific, Int, SpecificInts);

template <typename T> class UInt : public ::testing::Test {};
TYPED_TEST_CASE_P(UInt);
TYPED_TEST_P(UInt, types) {
  Type t = typeOf<TypeParam>();
  ASSERT_EQ(Type::UInt, t.getKind());
  ASSERT_TRUE(t.isUInt());
  ASSERT_EQ(sizeof(TypeParam)*8, t.getNumBits());
  ASSERT_EQ(sizeof(TypeParam), t.getNumBytes());
}
REGISTER_TYPED_TEST_CASE_P(UInt, types);
typedef ::testing::Types<unsigned char, unsigned short, unsigned int,
                         unsigned long, unsigned long long> GenericUInts;
INSTANTIATE_TYPED_TEST_CASE_P(Generic, UInt, GenericUInts);
typedef ::testing::Types<uint8_t, uint16_t, uint32_t, uint64_t> SpecificFloat;
INSTANTIATE_TYPED_TEST_CASE_P(specific, UInt, SpecificFloat);

template <typename T> class Float : public ::testing::Test {};
TYPED_TEST_CASE_P(Float);
TYPED_TEST_P(Float, types) {
  Type t = typeOf<TypeParam>();
  ASSERT_EQ(Type::Float, t.getKind());
  ASSERT_TRUE(t.isFloat());
  ASSERT_EQ(sizeof(TypeParam)*8, t.getNumBits());
  ASSERT_EQ(sizeof(TypeParam), t.getNumBytes());
}
REGISTER_TYPED_TEST_CASE_P(Float, types);
typedef ::testing::Types<float, double> GenericFloat;
INSTANTIATE_TYPED_TEST_CASE_P(Generic, Float, GenericFloat);
