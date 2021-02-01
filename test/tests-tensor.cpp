#include "test.h"
#include "taco/component.h"
#include "taco/tensor.h"
#include "test_tensors.h"

#include <sstream>
#include <string>
#include <vector>
#include "taco/util/collections.h"

using namespace taco;

template<typename T>
void testFill(T fillVal) {
  Tensor<T> a({2,2}, fillVal);
  ASSERT_TRUE(equals(a.getFillValue(), Literal((T) fillVal)));
}

TEST(tensor, double_scalar) {
  Tensor<double> a(4.2);
  ASSERT_DOUBLE_EQ(4.2, a.begin()->second);
}

TEST(tensor, double_vector) {
  Tensor<double> a({5}, Sparse);
  ASSERT_EQ(Float64, a.getComponentType());
  ASSERT_EQ(1, a.getOrder());
  ASSERT_EQ(5, a.getDimension(0));

  ASSERT_TRUE(a.needsPack());

  map<vector<int>,double> vals = {{{0}, 1.0}, {{2}, 2.0}};
  for (auto& val : vals) {
    a.insert(val.first, val.second);
  }
  ASSERT_TRUE(a.needsPack());
  a.pack();
  ASSERT_FALSE(a.needsPack());

  for (auto val = a.beginTyped<int>(); val != a.endTyped<int>(); ++val) {
    ASSERT_TRUE(util::contains(vals, val->first.toVector()));
    ASSERT_EQ(vals.at(val->first.toVector()), val->second);
  }

  TensorBase abase = a;

  for (auto val = abase.iteratorTyped<int, double>().begin(); val != abase.iteratorTyped<int, double>().end(); ++val) {
    ASSERT_TRUE(util::contains(vals, val->first.toVector()));
    ASSERT_EQ(vals.at(val->first.toVector()), val->second);
  }
}

TEST(tensor, iterate) {
  Tensor<double> a({5}, Sparse);
  a.insert({1}, 10.0);
  a.pack();
  ASSERT_TRUE(a.begin() != a.end());
  ASSERT_TRUE(++a.begin() == a.end());
  ASSERT_DOUBLE_EQ(10.0, a.begin()->second);
}

TEST(tensor, iterate_empty) {
  Tensor<double> a({5}, Sparse);
  a.pack();
  ASSERT_TRUE(a.begin() == a.end());
}

TEST(tensor, duplicates) {
  Tensor<double> a({5,5}, Sparse);
  a.insert({1,2}, 42.0);
  a.insert({2,2}, 10.0);
  a.insert({1,2}, 1.0);
  a.pack();
  map<vector<int>,double> vals = {{{1,2}, 43.0}, {{2,2}, 10.0}};
  for (auto val = a.beginTyped<int>(); val != a.endTyped<int>(); ++val) {
    ASSERT_TRUE(util::contains(vals, val->first.toVector()));
    ASSERT_EQ(vals.at(val->first.toVector()), val->second);
  }
}

TEST(tensor, duplicates_scalar) {
  Tensor<double> a;
  a.insert({}, 1.0);
  a.insert({}, 2.0);
  a.pack();
  auto val = a.begin();
  ASSERT_EQ(val->second, 3.0);
  ASSERT_TRUE(++val == a.end());
}

TEST(tensor, scalar_type_correct) {
  Tensor<int> a;
  ASSERT_EQ(a.getComponentType(), Int32);
}

TEST(tensor, non_zero_fill) {
  testFill<char>(3);
  testFill<int>(34762);
  testFill<uint16_t>((1 << 10));
  testFill<uint32_t>((1 << 30));
  testFill<uint64_t>((1ULL << 42));
  testFill<int16_t>((1 << 10));
  testFill<int32_t>(-1);
  testFill<int64_t>((1ULL << 42));
  testFill<float>(std::numeric_limits<float>::min());
  testFill<double>(std::numeric_limits<double>::max());
  testFill<double>(std::numeric_limits<double>::infinity());
}

TEST(tensor, transpose) {
  TensorData<double> testData = TensorData<double>({5, 3, 2}, {
    {{0,0,0}, 0.0},
    {{0,0,1}, 1.0},
    {{0,1,0}, 2.0},
    {{0,1,1}, 3.0},
    {{2,0,0}, 4.0},
    {{2,0,1}, 5.0},
    {{4,0,0}, 6.0},
  });
  TensorData<double> transposedTestData = TensorData<double>({2, 5, 3}, {
    {{0,0,0}, 0.0},
    {{1,0,0}, 1.0},
    {{0,0,1}, 2.0},
    {{1,0,1}, 3.0},
    {{0,2,0}, 4.0},
    {{1,2,0}, 5.0},
    {{0,4,0}, 6.0},
  });

  Tensor<double> tensor = testData.makeTensor("a", Format({Sparse, Dense, Sparse}, {1, 0, 2}));
  tensor.pack();
  Tensor<double> transposedTensor = transposedTestData.makeTensor("b", Format({Sparse, Dense, Sparse}, {1, 0, 2}));
  transposedTensor.pack();
  ASSERT_TRUE(equals(tensor.transpose({2,0,1}), transposedTensor));

  Tensor<double> transposedTensor2 = transposedTestData.makeTensor("b", Format({Sparse, Sparse, Dense}, {2, 1, 0}));
  transposedTensor2.pack();
  ASSERT_TRUE(equals(tensor.transpose({2,0,1}, Format({Sparse, Sparse, Dense}, {2, 1, 0})), transposedTensor2));
  ASSERT_TRUE(equals(tensor.transpose({0,1,2}), tensor));
}

TEST(tensor, operator_parens_insertion) {
  Tensor<double> a({5,5}, Sparse);
  a(1,2) = 42.0;
  a(2,2) = 10.0;
  a.pack();
  map<vector<int>,double> vals = {{{1,2}, 42.0}, {{2,2}, 10.0}};
  for (auto val = a.beginTyped<int>(); val != a.endTyped<int>(); ++val) {
    ASSERT_TRUE(util::contains(vals, val->first.toVector()));
    ASSERT_EQ(vals.at(val->first.toVector()), val->second);
  }
}

TEST(tensor, get_value) {
  Tensor<double> a({5,5}, Sparse);
  a(1,2) = 42.0;
  a(2,2) = 10.0;
  a.pack();

  double val1 = 42.0;
  double val2 = 10.0;

  ASSERT_EQ(val1, a.at({1,2}));
  ASSERT_EQ(val2, (double)a(2,2));
}

TEST(tensor, set_from_components) {
  typedef Component<2, double> C;

  std::vector<C> component_list;
  component_list.push_back(C({1,2}, 42.0));
  component_list.push_back(C({2,2}, 10.0));

  Tensor<double> a({5,5}, Sparse);
  a.setFromComponents(component_list.begin(), component_list.end());
  a.pack();

  map<vector<int>,double> vals = {{{1,2}, 42.0}, {{2,2}, 10.0}};
  for (auto val = a.beginTyped<int>(); val != a.endTyped<int>(); ++val) {
    ASSERT_TRUE(util::contains(vals, val->first.toVector()));
    ASSERT_EQ(vals.at(val->first.toVector()), val->second);
  }
}

TEST(tensor, hidden_pack) {
  Tensor<double> a({5,5}, Sparse);
  a(1,2) = 42.0;
  a(2,2) = 10.0;

  ASSERT_TRUE(a.needsPack());

  double val1 = 42.0;
  double val2 = 10.0;

  ASSERT_EQ(val1, a.at({1,2}));
  ASSERT_FALSE(a.needsPack());
  ASSERT_EQ(val2, (double)a(2,2));
}

TEST(tensor, automatic_pack_before_iteration) {
  Tensor<double> a({5,5}, Sparse);
  a(1,2) = 42.0;
  a(2,2) = 10.0;

  ASSERT_TRUE(a.needsPack());

  map<vector<int>,double> vals = {{{1,2}, 42.0}, {{2,2}, 10.0}};
  for (auto val = a.beginTyped<int>(); val != a.endTyped<int>(); ++val) {
    ASSERT_FALSE(a.needsPack());
    ASSERT_TRUE(util::contains(vals, val->first.toVector()));
    ASSERT_EQ(vals.at(val->first.toVector()), val->second);
  }
}

TEST(tensor, automatic_pack_before_const_iteration) {
  Tensor<double> a({5,5}, Sparse);
  a(1,2) = 42.0;
  a(2,2) = 10.0;

  const Tensor<double> b = a;
  map<vector<int>,double> vals = {{{1,2}, 42.0}, {{2,2}, 10.0}};
  for (auto val = b.begin(); val != b.end(); ++val) {
    ASSERT_TRUE(util::contains(vals, val->first.toVector()));
    ASSERT_EQ(vals.at(val->first.toVector()), val->second);
  }
}

TEST(tensor, hidden_compiler_methods) {
  Format csr({Dense,Sparse});
  Format csf({Sparse,Sparse,Sparse});
  Format  sv({Sparse});

  Tensor<double> A({2,3},   csr);
  Tensor<double> B({2,3,4}, csf);
  Tensor<double> c({4},     sv);

  B(0,0,0) = 1.0;
  B(1,2,0) = 2.0;
  B(1,2,1) = 3.0;
  c(0) = 4.0;
  c(1) = 5.0;

  ASSERT_TRUE(B.needsPack());
  ASSERT_TRUE(c.needsPack());

  IndexVar i, j, k;
  A(i,j) = B(i,j,k) * c(k);

  ASSERT_TRUE(A.needsCompile());
  ASSERT_TRUE(A.needsAssemble());
  ASSERT_TRUE(A.needsCompute());
  ASSERT_TRUE(B.needsPack());
  ASSERT_TRUE(c.needsPack());

  ASSERT_FALSE(A.needsPack());
  ASSERT_FALSE(B.needsCompile());
  ASSERT_FALSE(B.needsAssemble());
  ASSERT_FALSE(B.needsCompute());
  ASSERT_FALSE(c.needsCompile());
  ASSERT_FALSE(c.needsAssemble());
  ASSERT_FALSE(c.needsCompute());

  // Perform a read operation, such as printing to a stream.
  std::ostringstream stream;
  stream << A;

  ASSERT_FALSE(A.needsCompile());
  ASSERT_FALSE(A.needsAssemble());
  ASSERT_FALSE(A.needsCompute());
  ASSERT_FALSE(B.needsPack());
  ASSERT_FALSE(c.needsPack());

  ASSERT_FALSE(A.needsPack());
  ASSERT_FALSE(B.needsCompile());
  ASSERT_FALSE(B.needsAssemble());
  ASSERT_FALSE(B.needsCompute());
  ASSERT_FALSE(c.needsCompile());
  ASSERT_FALSE(c.needsAssemble());
  ASSERT_FALSE(c.needsCompute());

  map<vector<int>,double> vals = {{{0,0}, 4.0}, {{1,2}, 23.0}};
  for (auto val = A.beginTyped<int>(); val != A.endTyped<int>(); ++val) {
    ASSERT_TRUE(util::contains(vals, val->first.toVector()));
    ASSERT_EQ(vals.at(val->first.toVector()), val->second);
  }
}

TEST(tensor, explicit_compiler_methods) {
  Format csr({Dense,Sparse});
  Format csf({Sparse,Sparse,Sparse});
  Format  sv({Sparse});

  Tensor<double> A({2,3},   csr);
  Tensor<double> B({2,3,4}, csf);
  Tensor<double> c({4},     sv);

  B(0,0,0) = 1.0;
  B(1,2,0) = 2.0;
  B(1,2,1) = 3.0;
  c(0) = 4.0;
  c(1) = 5.0;

  ASSERT_TRUE(B.needsPack());
  ASSERT_TRUE(c.needsPack());

  B.pack();
  c.pack();

  ASSERT_FALSE(B.needsPack());
  ASSERT_FALSE(c.needsPack());

  ASSERT_FALSE(A.needsCompile());
  ASSERT_FALSE(A.needsAssemble());
  ASSERT_FALSE(A.needsCompute());

  IndexVar i, j, k;
  A(i,j) = B(i,j,k) * c(k);

  ASSERT_TRUE(A.needsCompile());
  ASSERT_TRUE(A.needsAssemble());
  ASSERT_TRUE(A.needsCompute());

  A.compile();
  ASSERT_FALSE(A.needsCompile());

  A.assemble();
  ASSERT_FALSE(A.needsAssemble());

  A.compute();
  ASSERT_FALSE(A.needsCompute());

  map<vector<int>,double> vals = {{{0,0}, 4.0}, {{1,2}, 23.0}};
  for (auto val = A.beginTyped<int>(); val != A.endTyped<int>(); ++val) {
    ASSERT_TRUE(util::contains(vals, val->first.toVector()));
    ASSERT_EQ(vals.at(val->first.toVector()), val->second);
  }
}

TEST(tensor, computation_dependency_modification) {
  Format csr({Dense,Sparse});
  Format csf({Sparse,Sparse,Sparse});
  Format  sv({Sparse});

  Tensor<double> A({2,3},   csr);
  Tensor<double> B({2,3,4}, csf);
  Tensor<double> c({4},     sv);

  B(0,0,0) = 1.0;
  B(1,2,0) = 2.0;
  B(1,2,1) = 3.0;
  c(0) = 4.0;
  c(1) = 5.0;

  ASSERT_TRUE(B.needsPack());
  ASSERT_TRUE(c.needsPack());

  IndexVar i, j, k;
  A(i,j) = B(i,j,k) * c(k);

  ASSERT_TRUE(A.needsCompile());
  ASSERT_TRUE(A.needsAssemble());
  ASSERT_TRUE(A.needsCompute());
  ASSERT_TRUE(B.needsPack());
  ASSERT_TRUE(c.needsPack());

  ASSERT_FALSE(A.needsPack());
  ASSERT_FALSE(B.needsCompile());
  ASSERT_FALSE(B.needsAssemble());
  ASSERT_FALSE(B.needsCompute());
  ASSERT_FALSE(c.needsCompile());
  ASSERT_FALSE(c.needsAssemble());
  ASSERT_FALSE(c.needsCompute());

  // Modify an operand of A
  c(0) = 1.0;

  ASSERT_FALSE(A.needsCompile());
  ASSERT_FALSE(A.needsAssemble());
  ASSERT_FALSE(A.needsCompute());
  ASSERT_FALSE(B.needsPack());
  ASSERT_TRUE(c.needsPack());

  ASSERT_FALSE(A.needsPack());
  ASSERT_FALSE(B.needsCompile());
  ASSERT_FALSE(B.needsAssemble());
  ASSERT_FALSE(B.needsCompute());
  ASSERT_FALSE(c.needsCompile());
  ASSERT_FALSE(c.needsAssemble());
  ASSERT_FALSE(c.needsCompute());

  map<vector<int>,double> vals = {{{0,0}, 4.0}, {{1,2}, 23.0}};
  for (auto val = A.beginTyped<int>(); val != A.endTyped<int>(); ++val) {
    ASSERT_TRUE(util::contains(vals, val->first.toVector()));
    ASSERT_EQ(vals.at(val->first.toVector()), val->second);
  }  
}

TEST(tensor, old_dependency_modification) {
  Format csr({Dense,Sparse});
  Format csf({Sparse,Sparse,Sparse});
  Format  sv({Sparse});

  Tensor<double> A({2,3},   csr);
  Tensor<double> B({2,3},   csr);
  Tensor<double> C({2,3},   csr);

  B(0,0) = 1.0;
  B(1,2) = 2.0;
  B(1,1) = 3.0;
  C(0,0) = 4.0;
  C(1,2) = 5.0;
  C(1,1) = 6.0;

  ASSERT_TRUE(B.needsPack());
  ASSERT_TRUE(C.needsPack());

  IndexVar i, j;
  A(i,j) = B(i,j);

  ASSERT_TRUE(A.needsCompile());
  ASSERT_TRUE(A.needsAssemble());
  ASSERT_TRUE(A.needsCompute());
  ASSERT_TRUE(B.needsPack());
  ASSERT_TRUE(C.needsPack());

  A(i,j) = C(i,j);

  ASSERT_TRUE(A.needsCompile());
  ASSERT_TRUE(A.needsAssemble());
  ASSERT_TRUE(A.needsCompute());
  ASSERT_TRUE(B.needsPack());
  ASSERT_TRUE(C.needsPack());

  // Modify an operand of A
  B(0,0) = 5.0;

  ASSERT_TRUE(A.needsCompile());
  ASSERT_TRUE(A.needsAssemble());
  ASSERT_TRUE(A.needsCompute());
  ASSERT_TRUE(B.needsPack());
  ASSERT_TRUE(C.needsPack());

  map<vector<int>,double> vals = {{{0,0}, 4.0}, {{1,2}, 5.0}, {{1,1}, 6.0}};
  for (auto val = A.beginTyped<int>(); val != A.endTyped<int>(); ++val) {
    ASSERT_TRUE(util::contains(vals, val->first.toVector()));
    ASSERT_EQ(vals.at(val->first.toVector()), val->second);
  }  
}

TEST(tensor, skip_recompile) {
  Tensor<double> a({3}, Format({Dense}));
  Tensor<double> b({3}, Format({Dense}));
  Tensor<double> c;
  
  a(0) = 4.0;
  a(1) = 5.0;
  a(2) = 6.0;

  IndexVar i;
  b(i) = a(i);
  c = b(i);

  ASSERT_TRUE(b.needsCompile());
  ASSERT_TRUE(c.needsCompile());
  ASSERT_EQ(c.begin()->second, 15);

  a(0) += 1.0;
  a(1) += 1.0;
  a(2) += 1.0;
  
  b(i) = a(i);
  c = b(i);
  
  ASSERT_FALSE(b.needsCompile());
  ASSERT_FALSE(c.needsCompile());
  ASSERT_EQ(c.begin()->second, 18);
}

TEST(tensor, recompile) {
  Tensor<double> a({3}, Format({Dense}));
  Tensor<double> b({3}, Format({Dense}));
  Tensor<double> c;
  
  a(0) = 4.0;
  a(1) = 5.0;
  a(2) = 6.0;

  IndexVar i;
  b(i) = a(i);
  c = b(i);

  ASSERT_TRUE(b.needsCompile());
  ASSERT_TRUE(c.needsCompile());
  ASSERT_EQ(c.begin()->second, 15.0);

  a(0) += 1.0;
  a(1) += 1.0;
  a(2) += 1.0;
  
  b(i) = a(i) + 1.0;
  c = 2.0 * b(i);
  
  ASSERT_TRUE(b.needsCompile());
  ASSERT_TRUE(c.needsCompile());
  ASSERT_EQ(c.begin()->second, 42.0);
}

TEST(tensor, cache) {
  auto dim = 2;
  IndexVar i("i"), j("j");
  Tensor<int> a("a", {dim, dim}, {Dense, Dense});
  Tensor<int> b("b", {dim, dim}, {Dense, Dense});
  Tensor<int> c("c", {dim, dim}, {Dense, Dense});
  // Add a computation to the cache.
  c(i, j) = a(i, j); c.evaluate();
  // Add a new computation to the cache.
  c(i, j) = a(i, j) + b(i, j); c.evaluate();
  // The addition of the new computation shouldn't have affected the cache's
  // ability to answer a request for the first query.
  c(i, j) = a(i, j); c.evaluate();
}
