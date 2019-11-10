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
const IndexVar i("i"), j("j"), k("k");

TEST(scheduling_eval, spmmCPU) {
  int NUM_I = 1021/10;
  int NUM_J = 1039/10;
  int NUM_K = 1057/10;
  float SPARSITY = .3;
  int UNROLL_FACTOR = 8;
  int CHUNK_SIZE = 16;
  Tensor<double> A("A", {NUM_I, NUM_J}, CSR);
  Tensor<double> B("B", {NUM_J, NUM_K}, {Dense, Dense});
  Tensor<double> C("C", {NUM_I, NUM_K}, {Dense, Dense});

  srand(0);
  for (int i = 0; i < NUM_I; i++) {
    for (int j = 0; j < NUM_J; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      if (rand_float < SPARSITY) {
        A.insert({i, j}, (double) ((int) (rand_float*3/SPARSITY)));
      }
    }
  }

  for (int j = 0; j < NUM_J; j++) {
    for (int k = 0; k < NUM_K; k++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      B.insert({j, k}, (double) ((int) (rand_float*3/SPARSITY)));
    }
  }

  A.pack();
  B.pack();

  IndexVar i("i"), j("j"), k("k");
  IndexVar i0("i0"), i1("i1"), jpos("jpos"), jpos0("jpos0"), jpos1("jpos1");
  C(i, k) = A(i, j) * B(j, k);

  IndexStmt stmt = C.getAssignment().concretize();
  stmt = stmt.split(i, i0, i1, CHUNK_SIZE)
          .pos(j, jpos, A(i,j))
          .split(jpos, jpos0, jpos1, UNROLL_FACTOR)
          .reorder({i0, i1, jpos0, k, jpos1})
          .parallelize(i0, PARALLEL_UNIT::CPU_THREAD, OUTPUT_RACE_STRATEGY::NO_RACES)
          .parallelize(k, PARALLEL_UNIT::CPU_VECTOR, OUTPUT_RACE_STRATEGY::IGNORE_RACES);

  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);
  ir::Stmt compute = lower(stmt, "compute",  false, true);
  codegen->compile(compute, true);

  C.compile(stmt);
  C.assemble();
  C.compute();

  Tensor<double> expected({NUM_I, NUM_K}, {Dense, Dense});
  expected(i, k) = A(i, j) * B(j, k);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, C);
}