#include "test.h"
#include "taco/tensor.h"
#include "taco/codegen/module.h"
#include "taco/index_notation/index_notation.h"
#include "taco/lower/lower.h"

using namespace taco;

// mixIndexing is a compilation test to ensure that we can index into a
// tensor with a mix of IndexVars and WindowedIndexVars.
TEST(windowing, mixIndexing) {
  auto dim = 10;
  Tensor<int> a("a", {dim, dim, dim, dim, dim}, {Dense, Dense, Dense, Dense, Dense});
  IndexVar i, j, k, l, m;
  auto w1 = a(i, j(1, 3), k, l(4, 5), m(6, 7));
  auto w2 = a(i(1, 3), j(2, 4), k, l, m(3, 5));
}

TEST(windowing, boundsChecks) {
  Tensor<int> a("a", {5}, {Dense});
  IndexVar i("i");
  ASSERT_THROWS_EXCEPTION_WITH_ERROR([&]() { a(i(-1, 4)); }, "slice lower bound");
  ASSERT_THROWS_EXCEPTION_WITH_ERROR([&]() { a(i(0, 10)); }, "slice upper bound");
}

// sliceMultipleWays tests that the same tensor can be sliced in different ways
// in the same expression.
TEST(windowing, sliceMultipleWays) {
  auto dim = 10;
  Tensor<int> a("a", {dim}, {Dense});
  Tensor<int> b("b", {dim}, {Sparse});
  Tensor<int> c("c", {dim}, {Dense});
  Tensor<int> expected("expected", {dim}, {Dense});
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

// basic tests a windowed tensor expression with different combinations
// of tensor formats.
TEST(windowing, basic) {
  Tensor<int> expectedAdd("expectedAdd", {2, 2}, {Dense, Dense});
  expectedAdd.insert({0, 0}, 14);
  expectedAdd.insert({0, 1}, 17);
  expectedAdd.insert({1, 0}, 17);
  expectedAdd.insert({1, 1}, 20);
  expectedAdd.pack();
  Tensor<int> expectedMul("expectedMul", {2, 2}, {Dense, Dense});
  expectedMul.insert({0, 0}, 64);
  expectedMul.insert({0, 1}, 135);
  expectedMul.insert({1, 0}, 135);
  expectedMul.insert({1, 1}, 240);
  expectedMul.pack();
  Tensor<int> d("d", {2, 2}, {Dense, Dense});

  // These dimensions are chosen so that one is above the constant in `mode_format_dense.cpp:54`
  // where the known stride is generated vs using the dimension.
  // TODO (rohany): Change that constant to be in a header file and import it here.
  for (auto& dim : {6, 20}) {
    for (auto &x : {Dense, Sparse}) {
      for (auto &y : {Dense, Sparse}) {
        for (auto &z : {Dense, Sparse}) {
          Tensor<int> a("a", {dim, dim}, {Dense, x});
          Tensor<int> b("b", {dim, dim}, {Dense, y});
          Tensor<int> c("c", {dim, dim}, {Dense, z});
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
      }
    }
  }
}

// slicedOutput tests that operations can write to a window within an output tensor.
TEST(windowing, slicedOutput) {
  auto dim = 10;
  Tensor<int> expected("expected", {10, 10}, {Dense, Dense});
  expected.insert({8, 8}, 12);
  expected.insert({8, 9}, 14);
  expected.insert({9, 8}, 14);
  expected.insert({9, 9}, 16);
  expected.pack();
  for (auto& x : {Dense, Sparse}) {
    for (auto& y : {Dense, Sparse}) {
      Tensor<int> a("a", {dim, dim}, {Dense, x});
      Tensor<int> b("b", {dim, dim}, {Dense, y});
      Tensor<int> c("c", {dim, dim}, {Dense, Dense});
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
  }
}

// transformations tests how windowing interacts with sparse iteration space
// transformations and different mode formats.
TEST(windowing, transformations) {
  auto dim = 10;
  Tensor<int> expected("expected", {2, 2}, {Dense, Dense});
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

  // TODO (rohany): Can we only reorder these loops in the Dense,Dense case? It seems so.
  testFn([&](IndexStmt stmt) {
    return stmt.reorder(i, j);
  }, {{Dense, Dense}});

  // We can only (currently) parallelize the outer dimension loop if it is dense.
  testFn([&](IndexStmt stmt) {
    return stmt.parallelize(i, taco::ParallelUnit::CPUThread, taco::OutputRaceStrategy::NoRaces);
  }, {{Dense, Dense}, {Dense, Sparse}});
}

// assignment tests assignments of and to windows in different combinations.
TEST(windowing, assignment) {
  auto dim = 10;

  auto testFn = [&](Format srcFormat) {
    Tensor<int> A("A", {dim, dim}, srcFormat);

    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        A.insert({i, j}, i + j);
      }
    }
    A.pack();

    IndexVar i, j;

    // First assign a window of A to a window of B.
    Tensor<int> B("B", {dim, dim}, {Dense, Dense});
    B(i(2, 4), j(3, 5)) = A(i(4, 6), j(5, 7));
    B.evaluate();
    Tensor<int> expected("expected", {dim, dim}, {Dense, Dense});
    expected.insert({2, 3}, 9); expected.insert({2, 4}, 10);
    expected.insert({3, 3}, 10); expected.insert({3, 4}, 11);
    expected.pack();
    ASSERT_TRUE(equals(B, expected)) << B << std::endl << expected << std::endl;

    // Assign a window of A to b.
    B = Tensor<int>("B", {2, 2}, {Dense, Dense});
    B(i, j) = A(i(4, 6), j(5, 7));
    B.evaluate();
    expected = Tensor<int>("expected", {2, 2}, {Dense, Dense});
    expected.insert({0, 0}, 9); expected.insert({0, 1}, 10);
    expected.insert({1, 0}, 10); expected.insert({1, 1}, 11);
    expected.pack();
    ASSERT_TRUE(equals(B, expected)) << B << std::endl << expected << std::endl;

    // Assign A to a window of B.
    A = Tensor<int>("A", {2, 2}, srcFormat);
    A.insert({0, 0}, 0); A.insert({0, 1}, 1);
    A.insert({1, 0}, 1); A.insert({1, 1}, 2);
    A.pack();
    B = Tensor<int>("B", {dim, dim}, {Dense, Dense});
    B(i(4, 6), j(5, 7)) = A(i, j);
    B.evaluate();
    expected = Tensor<int>("expected", {dim, dim}, {Dense, Dense});
    expected.insert({4, 5}, 0); expected.insert({4, 6}, 1);
    expected.insert({5, 5}, 1); expected.insert({5, 6}, 2);
    expected.pack();
    ASSERT_TRUE(equals(B, expected)) << B << std::endl << expected << std::endl;
  };

  for (auto& x : {Dense, Sparse}) {
    testFn({Dense, x});
  }
}

TEST(windowing, stride) {
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

  for (auto& x : {Dense, Sparse}) {
    for (auto& y : {Dense, Sparse}) {
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
    }
  }
}