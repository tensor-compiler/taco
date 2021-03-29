#include <taco/index_notation/transformations.h>
#include <codegen/codegen_c.h>
#include <codegen/codegen_cuda.h>
#include "test.h"
#include "test_tensors.h"
#include "taco/tensor.h"
#include "taco/index_notation/index_notation.h"
#include "codegen/codegen.h"
#include "taco/lower/lower.h"

using namespace taco;

TEST(blocking, test_dense_matrix_matrix_mult) {
 
  Tensor<float> A("A", {{8, 8}, {4, 4}},
                   Format({{Dense, Dense}, {Dense, Dense}}));
  Tensor<float> B("B", {{8, 8}, {4, 4}},
                   Format({{Dense, Dense}, {Dense, Dense}}));
  Tensor<float> C("C", {{8, 8}, {4, 4}},
                   Format({{Dense, Dense}, {Dense, Dense}}));

  Tensor<float> B1("B1", {8, 8, 4, 4},
                   Format({Dense, Dense, Dense, Dense}));
  Tensor<float> C1("C1", {8, 8, 4, 4},
                   Format({Dense, Dense, Dense, Dense}));

  for (int i0 = 0; i0 < 8; i0++) {
    for (int j0 = 0; j0 < 8; j0++) {
      for (int i1 = 0; i1 < 4; i1++) {
        for (int j1 = 0; j1 < 4; j1++) {
          float rand_float = (float)rand()/(float)(RAND_MAX);
          C1.insert({i0, j0, i1, j1}, rand_float);
          C.insert({{i0, j0}, {i1, j1}}, rand_float);
          // TODO: also support inserting matrices?
          rand_float = (float)rand()/(float)(RAND_MAX);
          B1.insert({i0, j0, i1, j1}, rand_float);
          B.insert({{i0, j0}, {i1, j1}}, rand_float);
        }
      }
    }
  }

  A.pack();
  B.pack();
  C.pack();
  B1.pack();
  C1.pack();

  IndexVar i0("i0"), j0("j0"), k0("k0");
  A(i0, k0) = B(i0, j0) * C(j0, k0);
  A.compile();
  A.assemble();
  A.compute();
  
  IndexVar i1("i1"), i2("i2"), j1("j1"), j2("j2"), k1("k1");
  Tensor<float> expected("expected", {8, 8, 4, 4},
                          Format({{Dense, Dense}, {Dense, Dense}}));

  expected(i1, k1, i2, j2) = B1(i1, j1, i2, j2) * C1(j1, k1, i2, j2);
  expected.compile();
  expected.assemble();
  expected.compute();

  for (int i0 = 0; i0 < 8; i0++) {
    for (int j0 = 0; j0 < 8; j0++) {
      Tensor<float> blocked_matrix = A(i0, j0);
      for (int i1 = 0; i1 < 4; i1++) {
        for (int j1 = 0; j1 < 4; j1++) {
          ASSERT_TRUE(expected(i0, j0, i1, j1) == blocked_matrix(i1, j1));
        }
      }
    }
  }
}
