#include "test.h"
#include "taco/tensor.h"
#include "taco/codegen/module.h"
#include "taco/index_notation/index_notation.h"
#include "taco/lower/lower.h"

using namespace taco;

// A basic test that filters values in a tensor with a simple predicate.
TEST(filtering, basic) {
  auto dim = 10;
  Tensor<int> a("a", {dim, dim}, {Dense, Sparse});
  Tensor<int> b("b", {dim, dim}, {Dense, Dense});
  Tensor<int> c("c", {dim, dim}, {Dense, Dense});
  Tensor<int> expected("expected", {dim, dim}, {Dense, Dense});

  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      a.insert({i, j}, i + j);
      b.insert({i, j}, i + j);
      if (i + j >= 5) {
        expected.insert({i, j}, 2 * (i + j));
      }
    }
  }
  a.pack(); b.pack(); expected.pack();

  // Apply a filter that elements must be larger than 5.
  auto filter = [](ir::Expr val) {
    return ir::Gte::make(val, ir::Literal::make(5));
  };

  IndexVar i("i"), j("j");
  c(i, j) = (a(i, j) | filter) + (b(i, j) | filter);
  c.evaluate();
  ASSERT_TRUE(equals(c, expected)) << c << endl << expected << endl;
}
