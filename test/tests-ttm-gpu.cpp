#include <taco/index_notation/transformations.h>
#include <codegen/codegen_c.h>
#include <codegen/codegen_cuda.h>
#include <fstream>
#include "test.h"
#include "test_tensors.h"
#include "taco/tensor.h"
#include "taco/index_notation/index_notation.h"
#include "codegen/codegen.h"
#include "taco/lower/lower.h"

using namespace taco;

const IndexVar i("i"), j("j"), k("k"), l("l");
IndexStmt schedule_ttm(IndexStmt stmt, Tensor<double> B, int NUM_L, int NNZ_PER_WARP=8*32,
                         int BLOCK_SIZE=256) {
  int WARP_SIZE = 32;
  int NNZ_PER_TB = NNZ_PER_WARP * (BLOCK_SIZE / WARP_SIZE);
  int CO_FACTOR = NUM_L / 32;
  assert(NUM_L % 32 == 0);
  IndexVar jk("jk"), f("f"), fpos("fpos"), block("block"),
          fpos1("fpos1"), warp("warp"), nnz("nnz"),
          dense_val_unbounded("dense_val_unbounded"),
          dense_val("dense_val"), thread("thread");
  return stmt.reorder({i, j, k, l})
          .fuse(j, k, jk)
          .fuse(i, jk, f)
          .pos(f, fpos, B(i, j, k))
          .split(fpos, block, fpos1, NNZ_PER_TB)
          .split(fpos1, warp, nnz, NNZ_PER_WARP)
          .split(l, dense_val_unbounded, thread, WARP_SIZE)
          .bound(dense_val_unbounded, dense_val, CO_FACTOR, BoundType::MaxExact)
          .reorder({block, warp, nnz, thread, dense_val})
          .unroll(dense_val, CO_FACTOR)
          .parallelize(block, ParallelUnit::GPUBlock, OutputRaceStrategy::IgnoreRaces)
          .parallelize(warp, ParallelUnit::GPUWarp, OutputRaceStrategy::IgnoreRaces)
          .parallelize(thread, ParallelUnit::GPUThread, OutputRaceStrategy::Atomics);
}

TEST(ttm_gpu, cpu) {
  // Create formats
  Format csf({Sparse,Sparse,Sparse});
  Format dense2d({Dense,Dense});

  // Create tensors
  int NUM_I = 128;
  int NUM_J = 128;
  int NUM_K = 128;
  int NUM_L = 128;
  Tensor<double> A("A", {NUM_I,NUM_J,NUM_L}, {Sparse,Sparse,Dense});
  Tensor<double> B("B", {NUM_I,NUM_J,NUM_K}, {Sparse,Sparse,Sparse});
  Tensor<double> C("C", {NUM_K,NUM_L}, dense2d);
  A(i,j,l) = B(i,j,k) * C(k,l);

  IndexStmt stmt = A.getAssignment().concretize();
  stmt = parallelizeOuterLoop(stmt);
  ir::Stmt compute = lower(stmt, "ttm_cpu",  false, true);

  stringstream source;
  ir::CodeGen_C codegen = ir::CodeGen_C(source, ir::CodeGen::ImplementationGen);
  codegen.compile(compute, true);

  ofstream source_file;
  source_file.open("ttm_cpu.c");
  source_file << source.str();
  source_file.close();
}

TEST(ttm_gpu, gpu) {
  // Create formats
  Format csf({Sparse,Sparse,Sparse});
  Format dense2d({Dense,Dense});

  // Create tensors
  int NUM_I = 128;
  int NUM_J = 128;
  int NUM_K = 128;
  int NUM_L = 128;
  Tensor<double> A("A",{NUM_I,NUM_J,NUM_L}, {Sparse,Sparse,Dense});
  Tensor<double> B("B", {NUM_I,NUM_J,NUM_K}, csf);
  Tensor<double> C("C", {NUM_K,NUM_L}, dense2d);

  vector<ModeFormat> Aformats = A.getFormat().getModeFormats();
  vector<ModeFormat> Bformats = B.getFormat().getModeFormats();
  for (int mode = 0; mode < 3; mode++) {
    assert(Aformats[mode] == Dense || Aformats[mode] == Bformats[mode]);
    // ensure simple sparse mask condition for ttm
    // (we support a more relaxed conditon for other kernels)
  }
  A(i,j,l) = B(i,j,k) * C(k,l);

  IndexStmt stmt = A.getAssignment().concretize();
  IndexStmt scheduled = schedule_ttm(stmt, B, NUM_L);
  ir::Stmt compute = lower(scheduled, "ttm_gpu",  false, true);

  stringstream source;
  ir::CodeGen_CUDA codegen = ir::CodeGen_CUDA(source, ir::CodeGen::ImplementationGen);
  codegen.compile(compute, true);

  ofstream source_file;
  source_file.open("ttm_gpu.cu");
  source_file << source.str();
  source_file.close();
}