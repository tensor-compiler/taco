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

TEST(blocking, test_dense_matrix_vector_mult) {

  // Tensor<double> A("A", {{8, 8}, {4, 4}},
  //                  Format({{Dense, Dense}, {Dense, Dense}}));
  // Tensor<double> B("B", {{8, 8}, {4, 4}},
  //                  Format({{Dense, Dense}, {Dense, Dense}}));
  // Tensor<double> C("C", {{8, 8}, {4, 4}},
  //                  Format({{Dense, Dense}, {Dense, Dense}}));

  Tensor<float> B("B", {8, 8, 4, 4},
                   Format({Dense, Dense, Dense, Dense}));
  Tensor<float> C("C", {8, 8, 4, 4},
                   Format({Dense, Dense, Dense, Dense}));

  for (int i0 = 0; i0 < 8; i0++) {
    for (int j0 = 0; j0 < 8; j0++) {
      for (int i1 = 0; i1 < 4; i1++) {
        for (int j1 = 0; j1 < 4; j1++) {
          float rand_float = (float)rand()/(float)(RAND_MAX);
          C.insert({i0, j0, i1, j1}, rand_float);
          // TODO: also support inserting matrices?
          rand_float = (float)rand()/(float)(RAND_MAX);
          B.insert({i0, j0, i1, j1}, rand_float);
        }
      }
    }
  }

  // A.pack();
  B.pack();
  C.pack();

  // A(i0, k0) = B(i0, j0) * C(j0, k0);

  IndexVar i0("i0"), i1("i1"), j0("j0"), j1("j1"), k0("k0");
  Tensor<float> expected("expected", {8, 8, 4, 4},
                          Format({{Dense, Dense}, {Dense, Dense}}));

  expected(i0, k0, i1, j1) = B(i0, j0, i1, j1) * C(j0, k0, i1, j1);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, expected);
}
