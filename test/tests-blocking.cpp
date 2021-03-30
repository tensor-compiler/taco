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

// TEST(blocking, test_dense_tv_format_only) {
  
//   int NUM_I = 1021/10;
//   int NUM_J = 1021/10;
//   int NUM_K = 1021/10;
//   int NUM_L = 1021/10;
  
//   Tensor<float> A("A", {NUM_K}, Format({ Dense }));
  
//   Tensor<float> B("B", {NUM_I, NUM_J, NUM_K, NUM_L},
//                    Format({{Dense, Dense}, {Dense, Dense}}));

//   Tensor<float> C("C", {NUM_L}, Format({ Dense }));

//   Tensor<float> B1("B1", {NUM_I, NUM_J, NUM_K, NUM_L},
//                    Format({Dense, Dense, Dense, Dense}));

//   for (int i0 = 0; i0 < NUM_I; i0++) {
//     for (int j0 = 0; j0 < NUM_J; j0++) {
//       for (int i1 = 0; i1 < NUM_K; i1++) {
//         for (int j1 = 0; j1 < NUM_L; j1++) {
//           float rand_float = (float)rand()/(float)(RAND_MAX);
//           B1.insert({i0, j0, i1, j1}, rand_float);
//           B.insert({i0, j0, i1, j1}, rand_float);
//         }
//       }
//     }
//   }

//   for (int i = 0; i < NUM_L; i++) {
//     float rand_float = (float)rand()/(float)(RAND_MAX);
//     C.insert({i}, rand_float);
//   }
   

//   A.pack();
//   B.pack();
//   C.pack();
//   B1.pack();

//   IndexVar i0("i0"), j0("j0"), k0("k0"), l0("k0");
//   A(k0) = B(i0, j0, k0, l0) * C(l0);
//   A.compile();
//   A.assemble();
//   A.compute();
  
//   IndexVar i1("i1"), j1("j1"), k1("k1"), l1("l1");
//   Tensor<float> expected("expected", {NUM_K}, Format({ Dense }));

//   expected(k1) = B1(i1, j1, k1, l1) * C(l1);
//   expected.compile();
//   expected.assemble();
//   expected.compute();

//   ASSERT_TENSOR_EQ(expected, A);
// }

TEST(blocking, test_sptv_format_only) {
  int NUM_I = 2;
  int NUM_J = 2;
  int NUM_K = 2;
  int NUM_L = 2;
  float SPARSITY = .01;
  
  Tensor<float> A("A", {NUM_K}, Format({ Dense }));
  
  Tensor<float> B("B", {NUM_I, NUM_J, NUM_K, NUM_L},
                   Format({{Sparse, Sparse}, {Sparse, Sparse}}));

  Tensor<float> C("C", {NUM_L}, Format({ Dense }));

  Tensor<float> B1("B1", {NUM_I, NUM_J, NUM_K, NUM_L},
                   Format({Sparse, Sparse, Sparse, Sparse}));

  srand(4357);
  for (int i0 = 0; i0 < NUM_I; i0++) {
    for (int j0 = 0; j0 < NUM_J; j0++) {
      for (int i1 = 0; i1 < NUM_K; i1++) {
        for (int j1 = 0; j1 < NUM_L; j1++) {
          float rand_float = (float)rand()/(float)(RAND_MAX);
          if (rand_float < SPARSITY) {
            B1.insert({i0, j0, i1, j1}, (float) ((int) (rand_float*3/SPARSITY)));
            B.insert({i0, j0, i1, j1}, (float) ((int) (rand_float*3/SPARSITY)));
          }
        }
      }
    }
  }


  for (int i = 0; i < NUM_L; i++) {
    float rand_float = (float)rand()/(float)(RAND_MAX);
    C.insert({i}, (float) ((int) (rand_float*3/SPARSITY)));
  }
   

  B.pack();
  // C.pack();
  // B1.pack();

  // IndexVar i0("i0"), j0("j0"), k0("k0"), l0("k0");
  // A(k0) = B(i0, j0, k0, l0) * C(l0);
  // A.compile();
  // A.assemble();
  // A.compute();
  
  // IndexVar i1("i1"), j1("j1"), k1("k1"), l1("l1");
  // Tensor<float> expected("expected", {NUM_K}, Format({ Dense }));

  // expected(k1) = B1(i1, j1, k1, l1) * C(l1);
  // expected.compile();
  // expected.assemble();
  // expected.compute();

  // ASSERT_TENSOR_EQ(expected, A);
}
