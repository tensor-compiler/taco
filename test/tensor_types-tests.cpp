#include "test.h"
#include "taco/type.h"
#include "test_tensors.h"
#include <math.h>

#include "taco/tensor.h"
#include "taco/storage/vector.h"

#define _USE_MATH_DEFINES


using namespace taco;
const IndexVar i("i"), j("j"), k("k");

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
  for (auto val = a.template beginTyped<int>(); val != a.template endTyped<int>(); ++val) {
    ASSERT_TRUE(util::contains(vals, val->first));
    ASSERT_EQ(vals.at(val->first), val->second);
  }
  
  TensorBase abase = a;
  Tensor<TypeParam> abaseIter = iterate<TypeParam>(a);
  for (auto val = abaseIter.template beginTyped<int>(); val != abaseIter.template endTyped<int>(); ++val) {
    ASSERT_TRUE(util::contains(vals, val->first));
    ASSERT_EQ(vals.at(val->first), val->second);
  }
}
REGISTER_TYPED_TEST_CASE_P(VectorTensorTest, types);
INSTANTIATE_TYPED_TEST_CASE_P(tensor_types, VectorTensorTest, AllTypes);


template <typename T> class IterateTensorTest : public ::testing::Test {};
TYPED_TEST_CASE_P(IterateTensorTest);
TYPED_TEST_P(IterateTensorTest, types) {
  Tensor<TypeParam> a({5}, Sparse);
  a.insert({1}, (TypeParam) 10.0);
  a.pack();
  ASSERT_TRUE(a.begin() != a.end());
  ASSERT_TRUE(++a.begin() == a.end());
  ASSERT_EQ((TypeParam) 10.0, (a.begin()++)->second);
}
REGISTER_TYPED_TEST_CASE_P(IterateTensorTest, types);
INSTANTIATE_TYPED_TEST_CASE_P(tensor_types, IterateTensorTest, AllTypes);


template <typename T> class AddTensorTest : public ::testing::Test {};
TYPED_TEST_CASE_P(AddTensorTest);
TYPED_TEST_P(AddTensorTest, types) {
  Tensor<TypeParam> a("a", {8}, Format({Sparse}, {0}));
  
  TensorData<TypeParam> testData = TensorData<TypeParam>({8}, {
    {{0}, (TypeParam) 10},
    {{2}, (TypeParam) 20},
    {{3}, (TypeParam) 30}
  });
  
  Tensor<TypeParam> b = testData.makeTensor("b", Format({Sparse}, {0}));
  b.pack();
  a(i) = b(i) + b(i);
  a.evaluate();
  
  Tensor<TypeParam> expected("a", {8}, Format({Sparse}, {0}));
  expected.insert({0}, (TypeParam) 20);
  expected.insert({2}, (TypeParam) 40);
  expected.insert({3}, (TypeParam) 60);
  expected.pack();
  DataType t = type<TypeParam>();

  ASSERT_TRUE(a.getComponentType() == t);
  //cout << "a: " << a;
  //cout << "expected: " << expected;
  ASSERT_TRUE(equals(expected,a));
}
REGISTER_TYPED_TEST_CASE_P(AddTensorTest, types);
INSTANTIATE_TYPED_TEST_CASE_P(tensor_types, AddTensorTest, AllTypes);


template <typename T> class MulTensorTest : public ::testing::Test {};
TYPED_TEST_CASE_P(MulTensorTest);
TYPED_TEST_P(MulTensorTest, types) {
  Tensor<TypeParam> a("a", {8}, Format({Sparse}, {0}));
  
  TensorData<TypeParam> testData = TensorData<TypeParam>({8}, {
    {{0}, (TypeParam) 10},
    {{2}, (TypeParam) 20},
    {{3}, (TypeParam) 30}
  });
  
  Tensor<TypeParam> b = testData.makeTensor("b", Format({Sparse}, {0}));
  b.pack();
  a(i) = b(i) * b(i);
  a.evaluate();
  
  Tensor<TypeParam> expected("a", {8}, Format({Sparse}, {0}));
  expected.insert({0}, (TypeParam) 100);
  expected.insert({2}, (TypeParam) 400);
  expected.insert({3}, (TypeParam) 900);
  expected.pack();
  DataType t = type<TypeParam>();
  ASSERT_TRUE(a.getComponentType() == t);
  ASSERT_TRUE(equals(expected,a));
}
REGISTER_TYPED_TEST_CASE_P(MulTensorTest, types);
INSTANTIATE_TYPED_TEST_CASE_P(tensor_types, MulTensorTest, AllTypes);


//Complex
TEST(tensor_types, complex_add) {
  Tensor<std::complex<float>> a("a", {8}, Format({Sparse}, {0}));
  
  TensorData<std::complex<float>> testData = TensorData<std::complex<float>>({8}, {
    {{0}, std::complex<float>(10.5, 10.5)},
    {{2}, std::complex<float>(20.5, 20.5)},
    {{3}, std::complex<float>(30.5, 30.5)},
  });
  
  Tensor<std::complex<float>> b = testData.makeTensor("b", Format({Sparse}, {0}));
  b.pack();
  a(i) = b(i) + b(i);
  a.evaluate();
  
  Tensor<std::complex<float>> expected("a", {8}, Format({Sparse}, {0}));
  expected.insert({0}, std::complex<float>(21, 21));
  expected.insert({2}, std::complex<float>(41, 41));
  expected.insert({3}, std::complex<float>(61, 61));
  expected.pack();
  
  ASSERT_TRUE(a.getComponentType() == Complex64);
  ASSERT_TRUE(equals(expected,a));
}

TEST(tensor_types, complex_mul_complex) {
  Tensor<std::complex<float>> a("a", {8}, Format({Sparse}, {0}));
  
  TensorData<std::complex<float>> testData = TensorData<std::complex<float>>({8}, {
    {{0}, std::complex<float>(10.5, 10.5)},
    {{2}, std::complex<float>(1, 0)},
    {{3}, std::complex<float>(0, 1)},
  });
  
  Tensor<std::complex<float>> b = testData.makeTensor("b", Format({Sparse}, {0}));
  b.pack();
  a(i) = b(i) * b(i);
  a.evaluate();
  
  Tensor<std::complex<float>> expected("a", {8}, Format({Sparse}, {0}));
  expected.insert({0}, std::complex<float>(0, 220.5));
  expected.insert({2}, std::complex<float>(1, 0));
  expected.insert({3}, std::complex<float>(-1, 0));
  expected.pack();
  
  ASSERT_TRUE(a.getComponentType() == Complex64);
  ASSERT_TRUE(equals(expected,a));
}

TEST(tensor_types, complex_mul_scalar) {
  Tensor<std::complex<float>> a("a", {8}, Format({Sparse}, {0}));
  
  TensorData<std::complex<float>> testData = TensorData<std::complex<float>>({8}, {
    {{0}, std::complex<float>(10.5, 10.5)},
    {{2}, std::complex<float>(1, 0)},
    {{3}, std::complex<float>(0, 1)},
  });
  
  Tensor<std::complex<float>> b = testData.makeTensor("b", Format({Sparse}, {0}));
  b.pack();
  Tensor<double> c(2.0);

  
  a(i) = c() * b(i);
  a.evaluate();
  
  Tensor<std::complex<float>> expected("a", {8}, Format({Sparse}, {0}));
  expected.insert({0}, std::complex<float>(21, 21));
  expected.insert({2}, std::complex<float>(2, 0));
  expected.insert({3}, std::complex<float>(0, 2));
  expected.pack();
  
  ASSERT_TRUE(a.getComponentType() == Complex64);
  ASSERT_TRUE(equals(expected,a));
}

TEST(tensor_types, complex_available_expr) {
  Tensor<std::complex<float>> a("a", {2, 2}, Format({Dense, Dense}));
  
  TensorData<std::complex<float>> testData = TensorData<std::complex<float>>({2, 2}, {
    {{0, 0}, std::complex<float>(0, 1)},
    {{0, 1}, std::complex<float>(1, 0)},
    {{1, 0}, std::complex<float>(1, 0)},
    {{1, 1}, std::complex<float>(1, 1)}
  });
  
  Tensor<std::complex<float>> b = testData.makeTensor("b", Format({Dense, Dense}));
  b.pack();
  
  TensorData<std::complex<float>> testData2 = TensorData<std::complex<float>>({2}, {
    {{0}, std::complex<float>(0, 1)},
    {{1}, std::complex<float>(1, 0)}
  });
  Tensor<std::complex<float>> c = testData2.makeTensor("c", Format({Dense}));
  c.pack();
  
  
  a(i, j) = b(i, j) * c(i);
  a.evaluate();
  
  Tensor<std::complex<float>> expected("a", {2, 2}, Format({Dense, Dense}));
  expected.insert({0, 0}, std::complex<float>(-1, 0));
  expected.insert({0, 1}, std::complex<float>(0, 1));
  expected.insert({1, 0}, std::complex<float>(1, 0));
  expected.insert({1, 1}, std::complex<float>(1, 1));
  expected.pack();
  
  ASSERT_TRUE(a.getComponentType() == Complex64);
  ASSERT_TRUE(equals(expected,a));
}

TEST(tensor_types, complex_accumulate) {
  Tensor<std::complex<float>> a("a", {2}, Format({Dense}));
  
  TensorData<std::complex<float>> testData = TensorData<std::complex<float>>({2, 2}, {
    {{0, 0}, std::complex<float>(0, 1)},
    {{0, 1}, std::complex<float>(1, 0)},
    {{1, 0}, std::complex<float>(1, 0)},
    {{1, 1}, std::complex<float>(1, 1)}
  });
  
  Tensor<std::complex<float>> b = testData.makeTensor("b", Format({Dense, Dense}));
  b.pack();
  
  TensorData<std::complex<float>> testData2 = TensorData<std::complex<float>>({2}, {
    {{0}, std::complex<float>(0, 1)},
    {{1}, std::complex<float>(1, 0)}
  });
  Tensor<std::complex<float>> c = testData2.makeTensor("c", Format({Dense}));
  c.pack();

  a(i) = b(i, j) * c(j);
  a.evaluate();
  
  Tensor<std::complex<float>> expected("a", {2}, Format({Dense}));
  expected.insert({0}, std::complex<float>(0, 0));
  expected.insert({1}, std::complex<float>(1, 2));
  expected.pack();
  
  ASSERT_TRUE(a.getComponentType() == Complex64);
  ASSERT_TRUE(equals(expected,a));
}

template <typename T>
bool equalsExact(Tensor<T> a, Tensor<T> b) {
  auto at = iterate<T>(a);
  auto bt = iterate<T>(b);
  auto ait = at.begin();
  auto bit = bt.begin();
  
  for (; ait != at.end() && bit != bt.end(); ++ait, ++bit) {
    if (ait->first != bit->first) {
      return false;
    }
    if (ait->second != bit->second) {
      return false;
    }
  }
  return (ait == at.end() && bit == bt.end());
}

TEST(tensor_types, float_double_promotion) {
  Tensor<double> a("a", {8}, Format({Sparse}, {0}));
  
  TensorData<float> testData = TensorData<float>({8}, {
    {{0}, (float) (M_PI)},
    {{2}, (float) (M_PI*2)},
    {{3}, (float) (M_PI*3)}
  });
  
  Tensor<float> b = testData.makeTensor("b", Format({Sparse}, {0}));
  b.pack();
  
  TensorData<double> testData2 = TensorData<double>({8}, {
    {{0}, (double) (M_PI*4)},
    {{2}, (double) (M_PI*5)},
    {{3}, (double) (M_PI*6)}
  });
  
  Tensor<double> c = testData2.makeTensor("c", Format({Sparse}, {0}));
  c.pack();
  
  a(i) = b(i) * c(i);
  a.evaluate();
  
  Tensor<double> expected("a", {8}, Format({Sparse}, {0}));
  expected.insert({0}, ((float) (M_PI)) * ((double) (M_PI*4)));
  expected.insert({2}, ((float) (M_PI*2)) * ((double) (M_PI*5)));
  expected.insert({3}, ((float) (M_PI*3)) * ((double) (M_PI*6)));
  expected.pack();
  
  Tensor<double> notexpected("a", {8}, Format({Sparse}, {0}));
  notexpected.insert({0}, (double)((float) (M_PI) * (float)(M_PI*4)));
  notexpected.insert({2}, (double)((float) (M_PI*2) * (float)(M_PI*5)));
  notexpected.insert({3}, (double)((float) (M_PI*3) * (float)(M_PI*6)));
  notexpected.pack();
  
  Tensor<double> notexpected2("a", {8}, Format({Sparse}, {0}));
  notexpected2.insert({0}, ((double) (M_PI)) * ((double) (M_PI*4)));
  notexpected2.insert({2}, ((double) (M_PI*2)) * ((double) (M_PI*5)));
  notexpected2.insert({3}, ((double) (M_PI*3)) * ((double) (M_PI*6)));
  notexpected2.pack();
  
  ASSERT_TRUE(equalsExact(a, expected));
  ASSERT_FALSE(equalsExact(a, notexpected));
  ASSERT_FALSE(equalsExact(a, notexpected2));
}

TEST(tensor_types, int_float_promotion) {
  Tensor<float> a("a", {8}, Format({Sparse}, {0}));
  
  TensorData<int> testData = TensorData<int>({8}, {
    {{0}, 1},
    {{2}, 2},
    {{3}, 3}
  });
  
  Tensor<int> b = testData.makeTensor("b", Format({Sparse}, {0}));
  b.pack();
  
  TensorData<float> testData2 = TensorData<float>({8}, {
    {{0}, .1f},
    {{2}, .2f},
    {{3}, .3f}
  });
  
  Tensor<float> c = testData2.makeTensor("c", Format({Sparse}, {0}));
  c.pack();
  
  a(i) = b(i) + c(i);
  a.evaluate();
  
  Tensor<float> expected("a", {8}, Format({Sparse}, {0}));
  expected.insert({0}, 1.1f);
  expected.insert({2}, 2.2f);
  expected.insert({3}, 3.3f);
  expected.pack();
  
  ASSERT_TRUE(equalsExact(a, expected));
}

TEST(tensor_types, coordinate_types) {
  TensorData<double> testData = TensorData<double>({5, 3, 2}, {
    {{0,0,0}, 0.0},
    {{0,0,1}, 1.0},
    {{0,1,0}, 2.0},
    {{0,1,1}, 3.0},
    {{2,0,0}, 4.0},
    {{2,0,1}, 5.0},
    {{4,0,0}, 6.0},
  });

  Format format = Format({Sparse, Dense, Sparse});
  format.setLevelArrayTypes({{Int32, Int16}, {Int32}, {Int64, Int16}});
  Tensor<double> tensor = testData.makeTensor("a", format);
  tensor.pack();

  int index = 0;
  for (std::pair<std::vector<size_t>,double> val : tensor) {
    std::vector<int> checkIndexes = testData.values[index].first;
    for (int coordIndex = 0; coordIndex < (int) checkIndexes.size(); coordIndex++) {
      ASSERT_EQ((size_t) checkIndexes[coordIndex], val.first[coordIndex]);
    }
    ASSERT_EQ(val.second, testData.values[index].second);
    index++;
  }

  Format format2 = Format({Sparse, Dense, Sparse});
  format.setLevelArrayTypes({{UInt8, Int16}, {UInt8}, {UInt64, Int32}});
  Tensor<double> tensor2 = testData.makeTensor("a", format2);
  tensor2.pack();

  index = 0;
  for (std::pair<std::vector<size_t>,double> val : tensor2) {
    std::vector<int> checkIndexes = testData.values[index].first;
    for (int coordIndex = 0; coordIndex < (int) checkIndexes.size(); coordIndex++) {
      ASSERT_EQ((size_t) checkIndexes[coordIndex], val.first[coordIndex]);
    }
    ASSERT_EQ(val.second, testData.values[index].second);
    index++;
  }

}

/*TEST(sparse_transpose, test) {
  Format ss({Sparse, Sparse}, {0, 1});
  Format sst({Sparse, Sparse}, {1, 0});
  TensorData<int> testData = TensorData<int>({3, 4}, {
    {{0, 0}, 6},
    {{0, 2}, 9},
    {{0, 3}, 8},
    {{2, 0}, 5},
    {{2, 3}, 7}
  });

  Tensor<int> a = testData.makeTensor("a", ss);
  a.pack();

  int row_ptr[] = {0, 2};
  int row_idx[] = {0, 2};
  int col_ptr[] = {0, 3, 5};
  int col_idx[] = {0, 2, 3, 0, 3};
  int vals[] = {6, 9, 8, 5, 7};

  int numpercol[4] = {0}; //num_col
  int numnonzerocols = 0;
  for (int i = 0; i < 5; i++) { //len col_idx
    numpercol[col_idx[i]]++;
  }
  for (int i = 0; i < 4; i++) { //num_col
    if (numpercol[i] > 0) numnonzerocols++;
  }

  cout << "transposed col_ptr: [0, " << numnonzerocols << "]" << endl;
  int transposed_col_idx[3]; //numnonzerocols
  int transposed_row_ptr[3+1]; //numnonzerocols + 1
  int transposed_row_idx[5]; //num values
  int transposed_vals[5]; //num values

  int row_ptr_val = 0;
  int col = 0;
  for (int i = 0; i < numnonzerocols; i++) {
    while (numpercol[col] == 0 && col < 4) { //num cols
      col++;
    }
    transposed_col_idx[i] = col;
    transposed_row_ptr[i] = row_ptr_val;
    row_ptr_val += numpercol[col];
    col++;
  }
  transposed_row_ptr[numnonzerocols] = row_ptr_val;

  cout << "transposed col_idx: [";
  for (int i = 0; i < 3; i++) {
    cout << transposed_col_idx[i] << ", ";
  }
  cout << "]" << endl;

  cout << "transposed row_ptr: [";
  for (int i = 0; i < 3+1; i++) {
    cout << transposed_row_ptr[i] << ", ";
  }
  cout << "]" << endl;

  //Iterate over matrix and scatter into row_idx and vals

  cout << endl << endl << "A: " << a << endl << endl;
  Tensor<int> at = testData.makeTensor("at", sst);
  at.pack();
  cout << "AT: " << at << endl << endl;


  Format sss({Sparse, Sparse, Sparse}, {0, 1, 2});
  Format sss01({Sparse, Sparse, Sparse}, {1, 0, 2});
  Format sss12({Sparse, Sparse, Sparse}, {0, 2, 1});
  Format sss02({Sparse, Sparse, Sparse}, {2, 1, 0});
  TensorData<int> testData3 = TensorData<int>({3, 3, 4}, {
    {{0, 0, 0}, 0},
    {{0, 0, 2}, 1},
    {{0, 0, 3,}, 2},
    {{0, 2, 0}, 3},
    {{0, 2, 3}, 4},
    {{2, 0, 0}, 10},
    {{2, 0, 2}, 11},
    {{2, 0, 3,}, 12},
    {{2, 2, 0}, 13},
    {{2, 2, 3}, 14}
  });

  Tensor<int> b = testData3.makeTensor("b", sss);
  b.pack();
  cout << endl << endl << "B: " << b << endl << endl;

  Tensor<int> b12 = testData3.makeTensor("b12", sss12);
  b12.pack();
  cout << "B12: " << b12 << endl << endl;

  Tensor<int> b01 = testData3.makeTensor("b01", sss01);
  b01.pack();
  cout << "B01: " << b01 << endl << endl;

  Tensor<int> b02 = testData3.makeTensor("b02", sss02);
  b02.pack();
  cout << "B02: " << b02 << endl << endl;
}*/

