#include "test.h"
#include "taco/tensor.h"
#include "taco/codegen/module.h"
#include "taco/index_notation/index_notation.h"
#include "taco/lower/lower.h"

#include <tuple>

using namespace taco;

// mixIndexing is a compilation test to ensure that we can index into a
// tensor with a mix of IndexVars and WindowedIndexVars.
TEST(windowing, mixIndexing) {
  auto dim = 10;
  Tensor<int> a("a", {dim, dim, dim, dim, dim}, Format{Dense, Dense, Dense, Dense, Dense});
  IndexVar i, j, k, l, m;
  auto w1 = a(i, j(1, 3), k, l(4, 5), m(6, 7));
  auto w2 = a(i(1, 3), j(2, 4), k, l, m(3, 5));
}

TEST(windowing, boundsChecks) {
  Tensor<int> a("a", {5}, Format{Dense});
  IndexVar i("i");
  ASSERT_THROWS_EXCEPTION_WITH_ERROR([&]() { a(i(-1, 4)); }, "slice lower bound");
  ASSERT_THROWS_EXCEPTION_WITH_ERROR([&]() { a(i(0, 10)); }, "slice upper bound");
}

// sliceMultipleWays tests that the same tensor can be sliced in different ways
// in the same expression.
TEST(windowing, sliceMultipleWays) {
  auto dim = 10;
  Tensor<int> a("a", {dim}, Format{Dense});
  Tensor<int> b("b", {dim}, Format{Sparse});
  Tensor<int> c("c", {dim}, Format{Dense});
  Tensor<int> expected("expected", {dim}, Format{Dense});
  for (int i = 0; i < dim; i++) {
    a.insert({i}, i);
    b.insert({i}, i);
  }
  expected.insert({2}, 10);
  expected.insert({3}, 13);
  a.pack(); b.pack(); expected.pack();
  IndexVar i("i"), j("j");

  c(i(2, 4)) = a(i(5, 7)) + a(i(1, 3)) + b(i(4, 6));
  c.evaluate();
  ASSERT_TRUE(equals(expected, c));
}

// The test basic tests basic windowing behavior parameterized by a dimension
// of the input tensors and formats for each of the tensors in the computation.
struct basic : public TestWithParam<std::tuple<int, ModeFormat, ModeFormat, ModeFormat>> {};
TEST_P(basic, windowing){
  Tensor<int> expectedAdd("expectedAdd", {2, 2}, Format{Dense, Dense});
  expectedAdd.insert({0, 0}, 14);
  expectedAdd.insert({0, 1}, 17);
  expectedAdd.insert({1, 0}, 17);
  expectedAdd.insert({1, 1}, 20);
  expectedAdd.pack();
  Tensor<int> expectedMul("expectedMul", {2, 2}, Format{Dense, Dense});
  expectedMul.insert({0, 0}, 64);
  expectedMul.insert({0, 1}, 135);
  expectedMul.insert({1, 0}, 135);
  expectedMul.insert({1, 1}, 240);
  expectedMul.pack();
  Tensor<int> d("d", {2, 2}, Format{Dense, Dense});

  // The test is parameterized by a dimension, and formats for the different tensors.
  auto dim = std::get<0>(GetParam());
  auto x = std::get<1>(GetParam());
  auto y = std::get<2>(GetParam());
  auto z = std::get<3>(GetParam());
  Tensor<int> a("a", {dim, dim}, Format{Dense, x});
  Tensor<int> b("b", {dim, dim}, Format{Dense, y});
  Tensor<int> c("c", {dim, dim}, Format{Dense, z});
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      a.insert({i, j}, i + j);
      b.insert({i, j}, i + j);
      c.insert({i, j}, i + j);
    }
  }

  a.pack();
  b.pack();
  c.pack();

  IndexVar i, j;
  d(i, j) = a(i(2, 4), j(2, 4)) + b(i(4, 6), j(4, 6)) + c(i(1, 3), j(1, 3));
  d.evaluate();
  ASSERT_TRUE(equals(expectedAdd, d))
                << endl << expectedAdd << endl << endl << d << endl
                << dim << " " << x << " " << y << " " << z << endl;

  d(i, j) = a(i(2, 4), j(2, 4)) * b(i(4, 6), j(4, 6)) * c(i(1, 3), j(1, 3));
  d.evaluate();
  ASSERT_TRUE(equals(expectedMul, d))
                << endl << expectedMul << endl << endl << d << endl
                << dim << " " << x << " " << y << " " << z << endl;
}
INSTANTIATE_TEST_CASE_P(
    windowing,
    basic,
    // Test on the cartesian product of the chosen dimensions and different
    // combinations for tensor formats.
    Combine(Values(6, 20), Values(Dense, Sparse), Values(Dense, Sparse), Values(Dense, Sparse))
);

// slicedOutput tests that operations can write to a window within an output tensor.
// The test is parameterized over formats for the used tensors.
struct slicedOutput : public TestWithParam<std::tuple<ModeFormat, ModeFormat>> {};
TEST_P(slicedOutput, windowing) {
  auto dim = 10;
  Tensor<int> expected("expected", {10, 10}, Format{Dense, Dense});
  expected.insert({8, 8}, 12);
  expected.insert({8, 9}, 14);
  expected.insert({9, 8}, 14);
  expected.insert({9, 9}, 16);
  expected.pack();
  auto x = std::get<0>(GetParam());
  auto y = std::get<1>(GetParam());
  Tensor<int> a("a", {dim, dim}, Format{Dense, x});
  Tensor<int> b("b", {dim, dim}, Format{Dense, y});
  Tensor<int> c("c", {dim, dim}, Format{Dense, Dense});
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      a.insert({i, j}, i + j);
      b.insert({i, j}, i + j);
    }
  }
  a.pack();
  b.pack();

  IndexVar i, j;
  c(i(8, 10), j(8, 10)) = a(i(2, 4), j(2, 4)) + b(i(4, 6), j(4, 6));
  c.evaluate();
  ASSERT_TRUE(equals(expected, c))
                << endl << expected << endl << endl << c << endl
                << dim << " " << x << " " << y << endl;
}
INSTANTIATE_TEST_CASE_P(
    windowing,
    slicedOutput,
    Combine(Values(Dense, Sparse), Values(Dense, Sparse))
);

// matrixMultiple tests a matrix multiply, and in the process is testing
// windowing on expressions that contain reductions. The test is parameterized
// over formats for the used tensors.
struct matrixMultiply : public TestWithParam<std::tuple<ModeFormat, ModeFormat>> {};
TEST_P(matrixMultiply, windowing) {
  auto dim = 10;
  auto windowDim = 4;

  Tensor<int> a("a", {windowDim, windowDim}, Format{Dense, Dense});
  Tensor<int> b("b", {windowDim, windowDim}, Format{Dense, Dense});
  Tensor<int> c("c", {windowDim, windowDim}, Format{Dense, Dense});
  Tensor<int> expected("expected", {windowDim, windowDim}, Format{Dense, Dense});

  auto x = std::get<0>(GetParam());
  auto y = std::get<1>(GetParam());
  Tensor<int> aw("aw", {dim, dim}, Format{Dense, x});
  Tensor<int> bw("bw", {dim, dim}, Format{Dense, y});
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      aw.insert({i, j}, i + j);
      bw.insert({i, j}, i + j);
    }
  }
  aw.pack(); bw.pack();

  IndexVar i("i"), j("j"), k("k");
  // Evaluate the windowed matrix multiply.
  c(i, k) = aw(i(4, 8), j(2, 6)) * bw(j(0, 4), k(6, 10));
  c.evaluate();

  // Copy the windowed portions of aw and bw into separate tensors, and test
  // that the un-windowed matrix multiplication has the same results.
  a(i, j) = aw(i(4, 8), j(2, 6));
  a.evaluate();
  b(i, j) = bw(i(0, 4), j(6, 10));
  b.evaluate();
  expected(i, k) = a(i, j) * b(j, k);
  expected.evaluate();

  ASSERT_TRUE(equals(expected, c)) << expected << endl << c << endl;
}
INSTANTIATE_TEST_CASE_P(
    windowing,
    matrixMultiply,
    Combine(Values(Dense, Sparse), Values(Dense, Sparse))
);

// workspace tests that workspaces can be assigned to and used in computations
// that involve windowed tensors. The test is parameterized over formats for
// the used tensors.
struct workspace : public TestWithParam<std::tuple<ModeFormat, ModeFormat>> {};
TEST_P(workspace, windowing) {
  auto dim = 10;
  size_t windowDim = 4;
  Tensor<int> d("d", {static_cast<int>(windowDim)}, Format{Dense});
  Tensor<int> expected("expected", {static_cast<int>(windowDim)}, Format{Dense});
  expected.insert({0}, 8); expected.insert({1}, 11);
  expected.insert({2}, 14); expected.insert({3}, 17);
  expected.pack();

  auto x = std::get<0>(GetParam());
  auto y = std::get<1>(GetParam());
  Tensor<int> a("a", {dim}, Format{x});
  Tensor<int> b("b", {dim}, Format{y});
  Tensor<int> c("c", {dim}, Format{Dense});
  for (int i = 0; i < dim; i++) {
    a.insert({i}, i);
    b.insert({i}, i);
    c.insert({i}, i);
  }
  a.pack();
  b.pack();
  c.pack();
  IndexVar i("i");
  TensorVar p("p", Type(Int(), {windowDim}), Dense);
  auto precomputed = a(i(2, 6)) + b(i(6, 10));
  d(i) = precomputed + c(i(0, 4));
  auto stmt = d.getAssignment().concretize();
  stmt = stmt.precompute(precomputed, i, i, p);
  d.compile(stmt.concretize());
  d.evaluate();
  ASSERT_TRUE(equals(d, expected)) << expected << endl << d << endl;
}
INSTANTIATE_TEST_CASE_P(
    windowing,
    workspace,
    Combine(Values(Dense, Sparse), Values(Dense, Sparse))
);

// transformations tests how windowing interacts with sparse iteration space
// transformations and different mode formats.
TEST(windowing, transformations) {
  auto dim = 10;
  Tensor<int> expected("expected", {2, 2}, Format{Dense, Dense});
  expected.insert({0, 0}, 12);
  expected.insert({0, 1}, 14);
  expected.insert({1, 0}, 14);
  expected.insert({1, 1}, 16);
  expected.pack();

  IndexVar i("i"), j("j"), i1 ("i1"), i2 ("i2");
  auto testFn = [&](std::function<IndexStmt(IndexStmt)> modifier, std::vector<Format> formats) {
    for (auto& format : formats) {
      Tensor<int> a("a", {dim, dim}, format);
      Tensor<int> b("b", {dim, dim}, format);
      for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
          a.insert({i, j}, i + j);
          b.insert({i, j}, i + j);
        }
      }
      a.pack(); b.pack();

      Tensor<int> c("c", {2, 2}, {Dense, Dense});
      c(i, j) = a(i(2, 4), j(2, 4)) + b(i(4, 6), j(4, 6));
      auto stmt = c.getAssignment().concretize();
      c.compile(modifier(stmt));
      c.evaluate();
      equals(c, expected);
      ASSERT_TRUE(equals(c, expected)) << endl << c << endl << expected << endl << format << endl;
    }
  };

  std::vector<Format> allFormats = {{Dense, Dense}, {Dense, Sparse}, {Sparse, Dense}, {Sparse, Sparse}};
  testFn([&](IndexStmt stmt) {
    return stmt.split(i, i1, i2, 4).unroll(i2, 4);
 }, allFormats);

  testFn([&](IndexStmt stmt) {
    return stmt.reorder(i, j);
  }, {{Dense, Dense}});

  // We can only (currently) parallelize the outer dimension loop if it is dense.
  testFn([&](IndexStmt stmt) {
    return stmt.parallelize(i, taco::ParallelUnit::CPUThread, taco::OutputRaceStrategy::NoRaces);
  }, {{Dense, Dense}, {Dense, Sparse}});
}

// assignment tests assignments of and to windows in different combinations.
// The test is parameterized over formats for the used tensors.
struct assignment : public TestWithParam<ModeFormat> {};
TEST_P(assignment, windowing) {
  auto dim = 10;
  auto srcFormat = GetParam();
  Tensor<int> A("A", {dim, dim}, srcFormat);

  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      A.insert({i, j}, i + j);
    }
  }
  A.pack();

  IndexVar i, j;

  // First assign a window of A to a window of B.
  Tensor<int> B("B", {dim, dim}, Format{Dense, Dense});
  B(i(2, 4), j(3, 5)) = A(i(4, 6), j(5, 7));
  B.evaluate();
  Tensor<int> expected("expected", {dim, dim}, Format{Dense, Dense});
  expected.insert({2, 3}, 9); expected.insert({2, 4}, 10);
  expected.insert({3, 3}, 10); expected.insert({3, 4}, 11);
  expected.pack();
  ASSERT_TRUE(equals(B, expected)) << B << std::endl << expected << std::endl;

  // Assign a window of A to b.
  B = Tensor<int>("B", {2, 2}, Format{Dense, Dense});
  B(i, j) = A(i(4, 6), j(5, 7));
  B.evaluate();
  expected = Tensor<int>("expected", {2, 2}, Format{Dense, Dense});
  expected.insert({0, 0}, 9); expected.insert({0, 1}, 10);
  expected.insert({1, 0}, 10); expected.insert({1, 1}, 11);
  expected.pack();
  ASSERT_TRUE(equals(B, expected)) << B << std::endl << expected << std::endl;

  // Assign A to a window of B.
  A = Tensor<int>("A", {2, 2}, srcFormat);
  A.insert({0, 0}, 0); A.insert({0, 1}, 1);
  A.insert({1, 0}, 1); A.insert({1, 1}, 2);
  A.pack();
  B = Tensor<int>("B", {dim, dim}, Format{Dense, Dense});
  B(i(4, 6), j(5, 7)) = A(i, j);
  B.evaluate();
  expected = Tensor<int>("expected", {dim, dim}, Format{Dense, Dense});
  expected.insert({4, 5}, 0); expected.insert({4, 6}, 1);
  expected.insert({5, 5}, 1); expected.insert({5, 6}, 2);
  expected.pack();
  ASSERT_TRUE(equals(B, expected)) << B << std::endl << expected << std::endl;
}
INSTANTIATE_TEST_CASE_P(
    windowing,
    assignment,
    Values(Dense, Sparse)
);

// cuda tests a basic windowing operation when using GPU targeted code.
// The test is parameterized over formats for the used tensors.
struct cuda : public TestWithParam<std::tuple<ModeFormat, ModeFormat>> {};
TEST_P(cuda, windowing) {
  if (!should_use_CUDA_codegen()) {
    return;
  }
  auto dim = 10;
  Tensor<int> expected("expected", {2, 2}, Format{Dense, Dense});
  expected.insert({0, 0}, 12); expected.insert({0, 1}, 14);
  expected.insert({1, 0}, 14); expected.insert({1, 1}, 16);
  expected.pack();

  auto x = std::get<0>(GetParam());
  auto y = std::get<1>(GetParam());
  Tensor<int> a("a", {dim, dim}, Format{Dense, x});
  Tensor<int> b("b", {dim, dim}, Format{Dense, y});
  Tensor<int> c("c", {2, 2}, Format{Dense, Dense});

  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      a.insert({i, j}, i + j);
      b.insert({i, j}, i + j);
    }
  }
  a.pack(); b.pack();

  IndexVar i("i"), j("j"), i1("i1"), i2("i2"), i3("i3"), i4("i4");
  c(i, j) = a(i(4, 6), j(4, 6)) + b(i(2, 4), j(2, 4));
  auto stmt = c.getAssignment().concretize();
  stmt = stmt.split(i, i1, i2, 512)
             .split(i2, i3, i4, 32)
             .parallelize(i1, ParallelUnit::GPUBlock, OutputRaceStrategy::NoRaces)
             .parallelize(i3, ParallelUnit::GPUWarp, OutputRaceStrategy::NoRaces)
             .parallelize(i4, ParallelUnit::GPUThread, OutputRaceStrategy::NoRaces);

  c.compile(stmt);
  c.evaluate();
  ASSERT_TRUE(equals(c, expected)) << c << endl << expected << endl;
}
INSTANTIATE_TEST_CASE_P(
    windowing,
    cuda,
    Combine(Values(Dense, Sparse), Values(Dense, Sparse))
);

// stride tests that tensor windows can be accessed with a stride. It is parameterized
// by formats of the used tensors.
struct stride : public TestWithParam<std::tuple<ModeFormat, ModeFormat>> {};
TEST_P(stride, windowing) {
  auto dim = 10;
  Tensor<int> expectedAdd("expectedAdd", {2, 2}, {Dense, Dense});
  expectedAdd.insert({0, 0}, 0); expectedAdd.insert({0, 1}, 10);
  expectedAdd.insert({1, 0}, 10); expectedAdd.insert({1, 1}, 20);
  expectedAdd.pack();
  Tensor<int> expectedAssign("expectedAssign", {2, 2}, {Dense, Dense});
  expectedAssign.insert({0, 0}, 0); expectedAssign.insert({0, 1}, 5);
  expectedAssign.insert({1, 0}, 5); expectedAssign.insert({1, 1}, 10);
  expectedAssign.pack();

  Tensor<int> expectedMul("expectedMul", {2, 2}, {Dense, Dense});
  expectedMul.insert({0, 0}, 0); expectedMul.insert({0, 1}, 25);
  expectedMul.insert({1, 0}, 25); expectedMul.insert({1, 1}, 100);
  expectedMul.pack();

  auto x = std::get<0>(GetParam());
  auto y = std::get<1>(GetParam());
  Tensor<int> a("a", {dim, dim}, {Dense, x});
  Tensor<int> b("b", {dim, dim}, {Dense, y});
  Tensor<int> c("c", {2, 2}, {Dense, Dense});
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      a.insert({i, j}, i+j);
      b.insert({i, j}, i+j);
    }
  }
  a.pack(); b.pack();

  IndexVar i("i"), j("j");

  // Test a strided assignment.
  c(i, j) = a(i(0, 10, 5), j(0, 10, 5));
  c.evaluate();
  ASSERT_TRUE(equals(c, expectedAssign)) << c << endl << expectedAssign << endl << x << " " << y << endl;

  // Test a strided addition.
  c(i, j) = a(i(0, 10, 5), j(0, 10, 5)) + b(i(0, 10, 5), j(0, 10, 5));
  c.evaluate();
  ASSERT_TRUE(equals(c, expectedAdd)) << c << endl << expectedAdd << endl;

  // Test a strided multiplication.
  c(i, j) = a(i(0, 10, 5), j(0, 10, 5)) * b(i(0, 10, 5), j(0, 10, 5));
  c.evaluate();
  ASSERT_TRUE(equals(c, expectedMul)) << c << endl << expectedMul << endl;

  // Test a strided assignment where the stride doesn't start at 0.
  c(i, j) = a(i(1, 5, 2), j(2, 6, 2));
  c.evaluate();
  Tensor<int> expectedAssign2("expectedAssign2", {2, 2}, {Dense, Dense});
  expectedAssign2.insert({0, 0}, 3); expectedAssign2.insert({0, 1}, 5);
  expectedAssign2.insert({1, 0}, 5); expectedAssign2.insert({1, 1}, 7);
  expectedAssign2.pack();
  ASSERT_TRUE(equals(c, expectedAssign2)) << c << endl << expectedAssign2 << endl;

}
INSTANTIATE_TEST_CASE_P(
    windowing,
    stride,
    Combine(Values(Dense, Sparse), Values(Dense, Sparse))
);

// indexSet tests that tensors can be windowed by a user provided set
// of elements. The test is parameterized by formats of the tested tensors.
struct indexSetVectors : public TestWithParam<std::tuple<ModeFormat, ModeFormat, ModeFormat>> {};
TEST_P(indexSetVectors, windowing) {
  auto dim = 10;
  auto x = std::get<0>(GetParam());
  auto y = std::get<1>(GetParam());
  auto z = std::get<2>(GetParam());

  Tensor<int> a("a", {3}, x);
  Tensor<int> b("b", {dim}, y);
  Tensor<int> c("c", {dim}, z);
  Tensor<int> expected("expected", {3}, Dense);
  for (int i = 0; i < dim; i++) {
    b.insert({i}, i);
    c.insert({i}, i);
  }
  expected.insert({0}, 1); expected.insert({1}, 9); expected.insert({2}, 25);
  b.pack(); c.pack(); expected.pack();

  IndexVar i("i");
  a(i) = b(i({1, 3, 5})) * c(i({1, 3, 5}));
  a.evaluate();
  ASSERT_TRUE(equals(a, expected)) << a << endl << expected << endl << x << " " << y << endl;

  expected = Tensor<int>("expected", {3}, Dense);
  expected.insert({0}, 2); expected.insert({1}, 6); expected.insert({2}, 10); expected.pack();
  a(i) = b(i({1, 3, 5})) + c(i({1, 3, 5}));
  a.evaluate();
  ASSERT_TRUE(equals(a, expected)) << a << endl << expected << endl << x << " " << y << endl;
}
INSTANTIATE_TEST_CASE_P(
    windowing,
    indexSetVectors,
    Combine(Values(Dense, Sparse), Values(Dense, Sparse), Values(Dense, Sparse))
);

struct indexSetMatrices : public TestWithParam<std::tuple<ModeFormat, ModeFormat, ModeFormat>> {};
TEST_P(indexSetMatrices, windowing) {
  auto dim = 10;
  auto x = std::get<0>(GetParam());
  auto y = std::get<1>(GetParam());
  auto z = std::get<2>(GetParam());

  Tensor<int> a("a", {3, 3}, {Dense, x});
  Tensor<int> expected("expected", {3, 3}, {Dense, Dense});
  Tensor<int> b("b", {dim, dim}, {Dense, y});
  Tensor<int> c("c", {dim, dim}, {Dense, z});
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      b.insert({i, j}, i + j);
      c.insert({i, j}, i + j);
    }
  }
  expected.insert({0, 0}, 10); expected.insert({0, 1}, 14); expected.insert({0, 2}, 18);
  expected.insert({1, 0}, 14); expected.insert({1, 1}, 18); expected.insert({1, 2}, 22);
  expected.insert({2, 0}, 18); expected.insert({2, 1}, 22); expected.insert({2, 2}, 26);
  b.pack(); c.pack(); expected.pack();

  IndexVar i("i"), j("j");
  a(i, j) = b(i({1, 3, 5}), j({2, 4, 6})) + c(i({3, 5, 7}), j({4, 6, 8}));
  a.compile();
  a.evaluate();
  ASSERT_TRUE(equals(a, expected)) << a << endl << expected << endl;
}
INSTANTIATE_TEST_CASE_P(
    windowing,
    indexSetMatrices,
    Combine(Values(Dense, Sparse), Values(Dense, Sparse), Values(Dense, Sparse))
);

// lhsIndexSet tests that assignments into tensors with indexSets.
TEST(windowing, lhsIndexSet) {
  auto dim = 10;
  Tensor<int> a("a", {dim}, Dense);
  Tensor<int> expected("expected", {dim}, Dense);
  Tensor<int> b("b", {3}, Dense);

  for (int i = 0; i < 3; i++) {
    b.insert({i}, i+1);
  }
  expected.insert({1}, 1); expected.insert({3}, 2); expected.insert({5}, 3);
  b.pack(); expected.pack();

  IndexVar i("i");
  a(i({1, 3, 5})) = b(i);
  a.compile();
  a.evaluate();
  ASSERT_TRUE(equals(a, expected)) << a << endl << expected << endl;
}

// See issue #451 for details about this test.
TEST(windowing, compoundAssign) {
  Tensor<double> A("A", {8}, Format{Dense});
  Tensor<double> B("B", {3}, Format{Dense});
  IndexVar i("i");
  A(i(0,3)) += B(i);
  // We explicitly call assemble here as the += makes evaluate not call it.
  A.compile(); A.assemble(); A.compute();
  A(i({1, 3, 6})) += B(i);
  A.evaluate();
}