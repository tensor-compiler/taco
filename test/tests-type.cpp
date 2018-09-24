#include "test.h"
#include "taco/type.h"

using namespace taco;
using namespace std;


template <typename T> class IntTest : public ::testing::Test {};
TYPED_TEST_CASE_P(IntTest);
TYPED_TEST_P(IntTest, types) {
  Datatype t = type<TypeParam>();
  ASSERT_TRUE(t.isInt());
  ASSERT_EQ(sizeof(TypeParam)*8, (size_t)t.getNumBits());
  ASSERT_EQ(sizeof(TypeParam),   (size_t)t.getNumBytes());
}
REGISTER_TYPED_TEST_CASE_P(IntTest, types);
typedef ::testing::Types<char, short, int, long, long long> GenericInts;
INSTANTIATE_TYPED_TEST_CASE_P(Generic, IntTest, GenericInts);
typedef ::testing::Types<int8_t, int16_t, int32_t, int64_t> SpecificInts;
INSTANTIATE_TYPED_TEST_CASE_P(specific, IntTest, SpecificInts);

template <typename T> class UIntTest : public ::testing::Test {};
TYPED_TEST_CASE_P(UIntTest);
TYPED_TEST_P(UIntTest, types) {
  Datatype t = type<TypeParam>();
  ASSERT_TRUE(t.isUInt());
  ASSERT_EQ(sizeof(TypeParam)*8, (size_t)t.getNumBits());
  ASSERT_EQ(sizeof(TypeParam),   (size_t)t.getNumBytes());
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
  Datatype t = type<TypeParam>();
  ASSERT_TRUE(t.isFloat());
  ASSERT_EQ(sizeof(TypeParam)*8, (size_t)t.getNumBits());
  ASSERT_EQ(sizeof(TypeParam),   (size_t)t.getNumBytes());
}
REGISTER_TYPED_TEST_CASE_P(FloatTest, types);
typedef ::testing::Types<float, double> GenericFloat;
INSTANTIATE_TYPED_TEST_CASE_P(Generic, FloatTest, GenericFloat);

TEST(type, equality) {
  Datatype fp32(Datatype::Float32);
  Datatype fp32_2(Datatype::Float32);
  Datatype fp64(Datatype::Float64);
  Datatype int32(Datatype::Int32);
  Datatype int64(Datatype::Int64);
  Datatype uint32(Datatype::UInt32);

  ASSERT_TRUE(fp32 == fp32);
  ASSERT_TRUE(fp32 == fp32_2);
  ASSERT_TRUE(!(fp32 == fp64));
  ASSERT_TRUE(fp32 != fp64);
  ASSERT_TRUE(fp32 != int32);
  ASSERT_TRUE(int32 != uint32);
  ASSERT_TRUE(int32 != uint32);
}

TEST(type, Dimension) {
  Dimension variable;
  ASSERT_TRUE(variable.isVariable());
  ASSERT_FALSE(variable.isFixed());
  ASSERT_EQ(Dimension(), variable);

  Dimension fixed(3);
  ASSERT_TRUE(fixed.isFixed());
  ASSERT_FALSE(fixed.isVariable());
  ASSERT_EQ(3u, fixed.getSize());
  ASSERT_EQ(Dimension(3), fixed);

  ASSERT_NE(variable, fixed);
}

TEST(type, Shape) {
  Dimension n, m, fixed(3);
  Shape shape({n,m,fixed,3});
  ASSERT_EQ(4, shape.getOrder());

  ASSERT_EQ(shape, Shape({n,m,fixed,3}));
  ASSERT_NE(shape, Shape({n,m,fixed,4}));
  ASSERT_NE(shape, Shape({n,m,Dimension(4),3}));
  ASSERT_NE(shape, Shape({n,3,fixed,3}));
}

TEST(type, TensorType) {
  Dimension n, m;
  Shape mn = {n,m};

  Type variable1(Float64, mn);
  ASSERT_EQ(2, variable1.getShape().getOrder());
  ASSERT_EQ(Type(Float64, mn), variable1);
  ASSERT_NE(Type(Float64, Shape({3,m})), variable1);

  Type variable2(Float64, {m,n});
  ASSERT_EQ(2, variable2.getShape().getOrder());
  ASSERT_EQ(variable1, variable2);

  Type fixed(Float64, {3,3});
  ASSERT_EQ(2, fixed.getShape().getOrder());
  ASSERT_EQ(3u, fixed.getShape().getDimension(0).getSize());
  ASSERT_EQ(Type(Float64, {3,3}), fixed);
  ASSERT_NE(Type(Float64, {3,3}), variable1);
}
