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
int WARP_SIZE = 32;

string file_path = "eval_generated/";
int status = mkdir(file_path.c_str(), 0777);

void printToCout(IndexStmt stmt) {
  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);
  ir::Stmt compute = lower(stmt, "compute", false, true);
  codegen->compile(compute, true);
}

void printToFile(string filename, IndexStmt stmt) {
  stringstream source;

  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(source, ir::CodeGen::ImplementationGen);
  ir::Stmt compute = lower(stmt, "compute",  false, true);
  codegen->compile(compute, true);

  ofstream source_file;
  string file_ending = should_use_CUDA_codegen() ? ".cu" : ".c";
  source_file.open(file_path + filename + file_ending);
  source_file << source.str();
  source_file.close();
}

IndexStmt scheduleSpMVCPU(IndexStmt stmt, int CHUNK_SIZE=16) {
  IndexVar i0("i0"), i1("i1"), kpos("kpos"), kpos0("kpos0"), kpos1("kpos1");
  return stmt.split(i, i0, i1, CHUNK_SIZE)
          .reorder({i0, i1, j})
          .parallelize(i0, PARALLEL_UNIT::CPU_THREAD, OUTPUT_RACE_STRATEGY::NO_RACES);
}

IndexStmt scheduleSpMMCPU(IndexStmt stmt, Tensor<double> A, int CHUNK_SIZE=16, int UNROLL_FACTOR=8) {
  IndexVar i0("i0"), i1("i1"), kbounded("kbounded"), k0("k0"), k1("k1"), jpos("jpos"), jpos0("jpos0"), jpos1("jpos1");
  return stmt.split(i, i0, i1, CHUNK_SIZE)
          .pos(j, jpos, A(i,j))
          .split(jpos, jpos0, jpos1, UNROLL_FACTOR)
          .reorder({i0, i1, jpos0, k, jpos1})
          .parallelize(i0, PARALLEL_UNIT::CPU_THREAD, OUTPUT_RACE_STRATEGY::NO_RACES)
          .parallelize(k, PARALLEL_UNIT::CPU_VECTOR, OUTPUT_RACE_STRATEGY::IGNORE_RACES);
}

IndexStmt scheduleSDDMMCPU(IndexStmt stmt, Tensor<double> B, int CHUNK_SIZE=16, int UNROLL_FACTOR=8) {
  IndexVar i0("i0"), i1("i1"), kpos("kpos"), kpos0("kpos0"), kpos1("kpos1");
  return stmt.split(i, i0, i1, CHUNK_SIZE)
          .pos(k, kpos, B(i,k))
          .split(kpos, kpos0, kpos1, UNROLL_FACTOR)
          .reorder({i0, i1, kpos0, j, kpos1})
          .parallelize(i0, PARALLEL_UNIT::CPU_THREAD, OUTPUT_RACE_STRATEGY::NO_RACES)
          .parallelize(kpos1, PARALLEL_UNIT::CPU_VECTOR, OUTPUT_RACE_STRATEGY::PARALLEL_REDUCTION);
}

IndexStmt scheduleTTVCPU(IndexStmt stmt, Tensor<double> B, int CHUNK_SIZE=16) {
  IndexVar f("f"), fpos("fpos"), chunk("chunk"), fpos2("fpos2");
  return stmt.fuse(i, j, f)
          .pos(f, fpos, B(i,j,k))
          .split(fpos, chunk, fpos2, CHUNK_SIZE)
          .reorder({chunk, fpos2, k})
          .parallelize(chunk, PARALLEL_UNIT::CPU_THREAD, OUTPUT_RACE_STRATEGY::NO_RACES);
}

IndexStmt scheduleTTMCPU(IndexStmt stmt, Tensor<double> B, int CHUNK_SIZE=16, int UNROLL_FACTOR=8) {
  IndexVar f("f"), fpos("fpos"), chunk("chunk"), fpos2("fpos2"), kpos("kpos"), kpos1("kpos1"), kpos2("kpos2");
  return stmt.fuse(i, j, f)
          .pos(f, fpos, B(i,j,k))
          .split(fpos, chunk, fpos2, CHUNK_SIZE)
          .pos(k, kpos, B(i,j,k))
          .split(kpos, kpos1, kpos2, UNROLL_FACTOR)
          .reorder({chunk, fpos2, kpos1, l, kpos2})
          .parallelize(chunk, PARALLEL_UNIT::CPU_THREAD, OUTPUT_RACE_STRATEGY::NO_RACES)
          .parallelize(kpos2, PARALLEL_UNIT::CPU_VECTOR, OUTPUT_RACE_STRATEGY::PARALLEL_REDUCTION);;
}

IndexStmt scheduleMTTKRPCPU(IndexStmt stmt, Tensor<double> B, int CHUNK_SIZE=16, int UNROLL_FACTOR=8) {
  IndexVar f("f"), fpos("fpos"), chunk("chunk"), fpos2("fpos2"), lpos("lpos"), lpos1("lpos1"), lpos2("lpos2");
  return stmt.reorder({i,k,j,l}) // TODO: this shouldn't be necessary
          .fuse(i, k, f)
          .pos(f, fpos, B(i,k,l))
          .split(fpos, chunk, fpos2, CHUNK_SIZE)
          .pos(l, lpos, B(i,k,l))
          .split(lpos, lpos1, lpos2, UNROLL_FACTOR)
          .reorder({chunk, fpos2, lpos1, j, lpos2})
          .parallelize(chunk, PARALLEL_UNIT::CPU_THREAD, OUTPUT_RACE_STRATEGY::ATOMICS)
          .parallelize(j, PARALLEL_UNIT::CPU_VECTOR, OUTPUT_RACE_STRATEGY::PARALLEL_REDUCTION);
}

IndexStmt scheduleSpMVGPU(IndexStmt stmt, Tensor<double> A, IndexExpr precomputedExpr, int NNZ_PER_THREAD=8, int BLOCK_SIZE=256) {
  int NNZ_PER_WARP = NNZ_PER_THREAD * WARP_SIZE;
  int NNZ_PER_TB = NNZ_PER_THREAD * BLOCK_SIZE;
  IndexVar f("f"), fpos("fpos"), fpos1("fpos1"), fpos2("fpos2"), block("block"), warp("warp"), thread("thread"), thread_nz("thread_nz"), thread_nz_pre("thread_nz_pre");
  TensorVar precomputed("precomputed", Type(Float64, {Dimension(thread_nz)}), taco::dense);
  return stmt.fuse(i, j, f)
          .pos(f, fpos, A(i, j))
          .split(fpos, block, fpos1, NNZ_PER_TB)
          .split(fpos1, warp, fpos2, NNZ_PER_WARP)
          .split(fpos2, thread, thread_nz, NNZ_PER_THREAD)
          .reorder({block, warp, thread, thread_nz})
          .precompute(precomputedExpr, thread_nz, thread_nz_pre, precomputed)
          .unroll(thread_nz_pre, NNZ_PER_THREAD)
          .parallelize(block, PARALLEL_UNIT::GPU_BLOCK, OUTPUT_RACE_STRATEGY::IGNORE_RACES)
          .parallelize(warp, PARALLEL_UNIT::GPU_WARP, OUTPUT_RACE_STRATEGY::IGNORE_RACES)
          .parallelize(thread, PARALLEL_UNIT::GPU_THREAD, OUTPUT_RACE_STRATEGY::ATOMICS);
}

IndexStmt scheduleSpMVRowsGPU(IndexStmt stmt, Tensor<double> A, IndexExpr precomputedExpr, int ROWS_PER_WARP=1, int BLOCK_SIZE=256) {
  int ROWS_PER_TB = ROWS_PER_WARP * BLOCK_SIZE;
  IndexVar block("block"), warp("warp"), thread("thread"), thread_nz("thread_nz"), i1("i1"), jpos("jpos"), block_row("block_row"), warp_row("warp_row");
  TensorVar precomputed("precomputed", Type(Float64, {Dimension(thread_nz)}), taco::dense);
  return stmt.split(i, block, block_row, ROWS_PER_TB)
          .split(block_row, warp_row, warp, BLOCK_SIZE / WARP_SIZE)
          .pos(j, jpos, A(i, j))
          .split(jpos, thread_nz, thread, WARP_SIZE)
          .reorder({block, warp, warp_row, thread, thread_nz})
          .parallelize(block, PARALLEL_UNIT::GPU_BLOCK, OUTPUT_RACE_STRATEGY::IGNORE_RACES)
          .parallelize(warp, PARALLEL_UNIT::GPU_WARP, OUTPUT_RACE_STRATEGY::IGNORE_RACES)
          .parallelize(thread, PARALLEL_UNIT::GPU_THREAD, OUTPUT_RACE_STRATEGY::TEMPORARY);
}

IndexStmt scheduleSpMVSplitPosGPU(IndexStmt stmt, Tensor<double> A, IndexExpr precomputedExpr, int NNZ_PER_THREAD=8, int BLOCK_SIZE=256) {
  int NNZ_PER_WARP = NNZ_PER_THREAD * WARP_SIZE;
  int NNZ_PER_TB = NNZ_PER_THREAD * BLOCK_SIZE;
  IndexVar f("f"), fpos("fpos"), fpos1("fpos1"), fpos2("fpos2"), block("block"), warp("warp"), thread("thread"), thread_nz("thread_nz"), thread_nz_pre("thread_nz_pre");
  TensorVar precomputed("precomputed", Type(Float64, {Dimension(thread_nz)}), taco::dense);
  return stmt.fuse(i, j, f)
          .pos(f, fpos, A(i, j))
          .split(fpos, block, fpos1, NNZ_PER_TB)
          .split(fpos1, warp, fpos2, NNZ_PER_WARP)
          .split(fpos2, thread, thread_nz, NNZ_PER_THREAD)
          .reorder({block, warp, thread, thread_nz})
          .parallelize(block, PARALLEL_UNIT::GPU_BLOCK, OUTPUT_RACE_STRATEGY::IGNORE_RACES)
          .parallelize(warp, PARALLEL_UNIT::GPU_WARP, OUTPUT_RACE_STRATEGY::IGNORE_RACES)
          .parallelize(thread, PARALLEL_UNIT::GPU_THREAD, OUTPUT_RACE_STRATEGY::ATOMICS);
}

IndexStmt scheduleSpMMGPU(IndexStmt stmt, Tensor<double> A, IndexExpr precomputedExpr, int NNZ_PER_WARP=8, int BLOCK_SIZE=256, int CO_FACTOR=4) {
  int NNZ_PER_TB = NNZ_PER_WARP * (BLOCK_SIZE / WARP_SIZE);
  IndexVar f("f"), fpos("fpos"), block("block"), fpos1("fpos1"), warp("warp"), nnz("nnz"), nnz_pre("nnz_pre");
  IndexVar dense_val_unbounded("dense_val_unbounded"), dense_val("dense_val"), thread("thread");
  IndexVar thread_nz("thread_nz");
  TensorVar precomputed("precomputed", Type(Float64, {Dimension(nnz)}), taco::dense);
  return stmt.reorder({i, j, k})
          .fuse(i, j, f)
          .pos(f, fpos, A(i, j))
          .split(fpos, block, fpos1, NNZ_PER_TB)
          .split(fpos1, warp, nnz, NNZ_PER_WARP)
          .split(k, dense_val_unbounded, thread, WARP_SIZE)
          .bound(dense_val_unbounded, dense_val, 1, BOUND_TYPE::MAX_EXACT)
          //.fuse(thread, dense_val, thread_nz)
          .reorder({block, warp, dense_val, thread, nnz})
          //.precompute(precomputedExpr, nnz, nnz, precomputed)
          //.unroll(dense_val, CO_FACTOR)
          .parallelize(block, PARALLEL_UNIT::GPU_BLOCK, OUTPUT_RACE_STRATEGY::IGNORE_RACES)
          .parallelize(warp, PARALLEL_UNIT::GPU_WARP, OUTPUT_RACE_STRATEGY::IGNORE_RACES)
          .parallelize(thread, PARALLEL_UNIT::GPU_THREAD, OUTPUT_RACE_STRATEGY::ATOMICS);
}

IndexStmt scheduleSDDMMGPU(IndexStmt stmt, Tensor<double> B, int NNZ_PER_WARP=8*32, int BLOCK_SIZE=256, int CO_FACTOR=4) {
  int NNZ_PER_TB = NNZ_PER_WARP * (BLOCK_SIZE / WARP_SIZE);
  IndexVar f("f"), fpos("fpos"), block("block"), fpos1("fpos1"), warp("warp"), nnz("nnz");
  IndexVar dense_val_unbounded("dense_val_unbounded"), dense_val("dense_val"), thread("thread");
  IndexVar thread_nz("thread_nz");
  return stmt.reorder({i, k, j})
          .fuse(i, k, f)
          .pos(f, fpos, B(i,k))
          .split(fpos, block, fpos1, NNZ_PER_TB)
          .split(fpos1, warp, nnz, NNZ_PER_WARP)
          .split(j, dense_val_unbounded, thread, WARP_SIZE)
          .bound(dense_val_unbounded, dense_val, CO_FACTOR, BOUND_TYPE::MAX_EXACT)
          .reorder({block, warp, nnz, thread, dense_val})
          .unroll(dense_val, CO_FACTOR)
          .parallelize(block, PARALLEL_UNIT::GPU_BLOCK, OUTPUT_RACE_STRATEGY::IGNORE_RACES)
          .parallelize(warp, PARALLEL_UNIT::GPU_WARP, OUTPUT_RACE_STRATEGY::ATOMICS)
          .parallelize(thread, PARALLEL_UNIT::GPU_THREAD, OUTPUT_RACE_STRATEGY::PARALLEL_REDUCTION);
}

IndexStmt scheduleTTMGPU(IndexStmt stmt, Tensor<double> B, int NNZ_PER_WARP=8*32, int BLOCK_SIZE=256, int CO_FACTOR=4) {
  int NNZ_PER_TB = NNZ_PER_WARP * (BLOCK_SIZE / WARP_SIZE);
  IndexVar jk("jk"), f("f"), fpos("fpos"), block("block"), fpos1("fpos1"), warp("warp"), nnz("nnz"), dense_val_unbounded("dense_val_unbounded"), dense_val("dense_val"), thread("thread");

  return stmt.reorder({i, j, k, l})
          .fuse(j, k, jk)
          .fuse(i, jk, f)
          .pos(f, fpos, B(i, j, k))
          .split(fpos, block, fpos1, NNZ_PER_TB)
          .split(fpos1, warp, nnz, NNZ_PER_WARP)
          .split(l, dense_val_unbounded, thread, WARP_SIZE)
          .bound(dense_val_unbounded, dense_val, CO_FACTOR, BOUND_TYPE::MAX_EXACT)
          .reorder({block, warp, nnz, thread, dense_val})
          .unroll(dense_val, CO_FACTOR)
          .parallelize(block, PARALLEL_UNIT::GPU_BLOCK, OUTPUT_RACE_STRATEGY::IGNORE_RACES)
          .parallelize(warp, PARALLEL_UNIT::GPU_WARP, OUTPUT_RACE_STRATEGY::IGNORE_RACES)
          .parallelize(thread, PARALLEL_UNIT::GPU_THREAD, OUTPUT_RACE_STRATEGY::ATOMICS);
}

IndexStmt scheduleTTVGPU(IndexStmt stmt, Tensor<double> B, IndexExpr precomputedExpr, int NNZ_PER_WARP=8*32, int BLOCK_SIZE=256) {
  int NNZ_PER_TB = NNZ_PER_WARP * (BLOCK_SIZE / WARP_SIZE);
  IndexVar jk("jk"), f("f"), fpos("fpos"), block("block"), fpos1("fpos1"), warp("warp"), fpos2("fpos2"), thread("thread"), thread_nz("thread_nz"), thread_nz_pre("thread_nz_pre");
  TensorVar precomputed("precomputed", Type(Float64, {Dimension(thread_nz)}), taco::dense);

  return stmt.fuse(j, k, jk)
          .fuse(i, jk, f)
          .pos(f, fpos, B(i,j,k))
          .split(fpos, block, fpos1, NNZ_PER_TB)
          .split(fpos1, warp, fpos2, NNZ_PER_WARP)
          .split(fpos2, thread, thread_nz, NNZ_PER_WARP/WARP_SIZE)
          .reorder({block, warp, thread, thread_nz})
          .precompute(precomputedExpr, thread_nz, thread_nz_pre, precomputed)
          .unroll(thread_nz_pre, NNZ_PER_WARP/WARP_SIZE)
          .parallelize(block, PARALLEL_UNIT::GPU_BLOCK, OUTPUT_RACE_STRATEGY::IGNORE_RACES)
          .parallelize(warp, PARALLEL_UNIT::GPU_WARP, OUTPUT_RACE_STRATEGY::IGNORE_RACES)
          .parallelize(thread, PARALLEL_UNIT::GPU_THREAD, OUTPUT_RACE_STRATEGY::ATOMICS);
}

IndexStmt scheduleMTTKRPGPU(IndexStmt stmt, Tensor<double> B, int NNZ_PER_WARP=8*32, int BLOCK_SIZE=256, int CO_FACTOR=4) {
  int NNZ_PER_TB = NNZ_PER_WARP * (BLOCK_SIZE / WARP_SIZE);
  IndexVar kl("kl"), f("f"), fpos("fpos"), block("block"), fpos1("fpos1"), warp("warp"), nnz("nnz"), dense_val_unbounded("dense_val_unbounded"), dense_val("dense_val"), thread("thread");
  return stmt.reorder({i,k,l,j})
          .fuse(k, l, kl)
          .fuse(i, kl, f)
          .pos(f, fpos, B(i, k, l))
          .split(fpos, block, fpos1, NNZ_PER_TB)
          .split(fpos1, warp, nnz, NNZ_PER_WARP)
          .split(j, dense_val_unbounded, thread, WARP_SIZE)
          .bound(dense_val_unbounded, dense_val, 1, BOUND_TYPE::MAX_EXACT)
          .reorder({block, warp, dense_val, thread, nnz})
          .parallelize(block, PARALLEL_UNIT::GPU_BLOCK, OUTPUT_RACE_STRATEGY::IGNORE_RACES)
          .parallelize(warp, PARALLEL_UNIT::GPU_WARP, OUTPUT_RACE_STRATEGY::IGNORE_RACES)
          .parallelize(thread, PARALLEL_UNIT::GPU_THREAD, OUTPUT_RACE_STRATEGY::ATOMICS);
}

// splits so same number of rows per warp and then each thread in warp gets 1/32 of the columns space
IndexStmt scheduleSpMMRowsGPU(IndexStmt stmt, Tensor<double> A, int ROWS_PER_WARP=4, int BLOCK_SIZE=256, int CO_FACTOR=4) {
  int ROWS_PER_TB = ROWS_PER_WARP * (BLOCK_SIZE / WARP_SIZE);
  IndexVar i1("i1"), block("block"), warp("warp"), warp_row("warp_row"), thread("thread"), thread_col("thread_col");
  return stmt.split(i, block, i1, ROWS_PER_TB)
          .split(i1, warp, warp_row, ROWS_PER_WARP)
          .split(k, thread, thread_col, 32)
          .reorder({block, warp, warp_row, thread, thread_col, j})
          .parallelize(block, PARALLEL_UNIT::GPU_BLOCK, OUTPUT_RACE_STRATEGY::IGNORE_RACES)
          .parallelize(warp, PARALLEL_UNIT::GPU_WARP, OUTPUT_RACE_STRATEGY::IGNORE_RACES)
          .parallelize(thread, PARALLEL_UNIT::GPU_THREAD, OUTPUT_RACE_STRATEGY::ATOMICS);
}

// splits so same number of nonzero rows per warp and then each thread in warp gets 1/32 of the columns space (no search needed)
IndexStmt scheduleSpMMNZRowsGPU(IndexStmt stmt, Tensor<double> A, int NZ_ROWS_PER_WARP=4, int BLOCK_SIZE=256, int CO_FACTOR=4) {
  int NZ_ROWS_PER_TB = NZ_ROWS_PER_WARP * (BLOCK_SIZE / WARP_SIZE);
  IndexVar ip("ip"), ip1("ip1"), block("block"), warp("warp"), warp_row("warp_row"), thread("thread"), thread_col("thread_col");
  return stmt.pos(i, ip, A(i, j))
          .split(ip, block, ip1, NZ_ROWS_PER_TB)
          .split(ip1, warp, warp_row, NZ_ROWS_PER_WARP)
          .split(k, thread, thread_col, 32)
          .reorder({block, warp, warp_row, thread, thread_col, j})
          .parallelize(block, PARALLEL_UNIT::GPU_BLOCK, OUTPUT_RACE_STRATEGY::IGNORE_RACES)
          .parallelize(warp, PARALLEL_UNIT::GPU_WARP, OUTPUT_RACE_STRATEGY::IGNORE_RACES)
          .parallelize(thread, PARALLEL_UNIT::GPU_THREAD, OUTPUT_RACE_STRATEGY::ATOMICS);
}


IndexStmt scheduleSpMMCPUNoVec(IndexStmt stmt, Tensor<double> A, int CHUNK_SIZE=16, int UNROLL_FACTOR=8) {
  IndexVar i0("i0"), i1("i1"), kbounded("kbounded"), k0("k0"), k1("k1"), jpos("jpos"), jpos0("jpos0"), jpos1("jpos1");
  return stmt.split(i, i0, i1, CHUNK_SIZE)
          .pos(j, jpos, A(i,j))
          .split(jpos, jpos0, jpos1, UNROLL_FACTOR)
          .reorder({i0, i1, jpos0, k, jpos1})
          .parallelize(i0, PARALLEL_UNIT::CPU_THREAD, OUTPUT_RACE_STRATEGY::NO_RACES);
}

IndexStmt exampleScheduleSPMVUntiled(IndexStmt stmt, Tensor<double> A) {
  return stmt;
}

IndexStmt exampleScheduleSPMVCPURowTiling(IndexStmt stmt, Tensor<double> A) {
  IndexVar i1("i1"), i2("i2");
  int ROWS_PER_TILE = 4;
  return stmt.split(i, i1, i2, ROWS_PER_TILE);
}

IndexStmt exampleScheduleSPMVPosIteration(IndexStmt stmt, Tensor<double> A) {
  IndexVar f("f"), p("p");
  return stmt.fuse(i, j, f)
             .pos(f, p, A(i, j));
}

TEST(scheduling_eval, test_spmvCPU_temp) {
  if (should_use_CUDA_codegen()) {
    return;
  }
  int NUM_I = 1021/10;
  int NUM_J = 1039/10;
  float SPARSITY = .3;
  Tensor<double> A("A", {NUM_I, NUM_J}, CSR);
  Tensor<double> x("x", {NUM_J}, {Dense});
  Tensor<double> y("y", {NUM_I}, {Dense});

  srand(4353);
  for (int i = 0; i < NUM_I; i++) {
    for (int j = 0; j < NUM_J; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      if (rand_float < SPARSITY) {
        A.insert({i, j}, (double) ((int) (rand_float*3/SPARSITY)));
      }
    }
  }

  for (int j = 0; j < NUM_J; j++) {
    float rand_float = (float)rand()/(float)(RAND_MAX);
    x.insert({j}, (double) ((int) (rand_float*3/SPARSITY)));
  }

  x.pack();
  A.pack();


  IndexVar i0("i0"), i1("i1"), kpos("kpos"), kpos0("kpos0"), kpos1("kpos1");
  TensorVar tj("tj", Float64);
  IndexVar jw("iw");

  y(i) = A(i, j) * x(j);
  Access tjAccess = tj();

  //IndexStmt stmt = forall(i, where(y(i) = tjAccess, forall(j, tjAccess += A(i, j) * x(j)))); //y.getAssignment().concretize();
  y(i) = A(i, j) * x(j);
  IndexStmt stmt = y.getAssignment().concretize();
  stmt = stmt.parallelize(i, PARALLEL_UNIT::CPU_THREAD, OUTPUT_RACE_STRATEGY::ATOMICS);

  printToFile("test_spmvCPU_temp", stmt);

  y.compile(stmt);
  y.assemble();
  y.compute();

  Tensor<double> expected("expected", {NUM_I}, {Dense});
  expected(i) = A(i, j) * x(j);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, y);
}

TEST(scheduling_eval, example_spmvCPU_splitpos) {
  if (should_use_CUDA_codegen()) {
    return;
  }
  int NUM_I = 1021/10;
  int NUM_J = 1039/10;
  float SPARSITY = .3;
  int CHUNK_SIZE = 16;
  Tensor<double> A("A", {NUM_I, NUM_J}, CSR);
  Tensor<double> x("x", {NUM_J}, {Dense});
  Tensor<double> y("y", {NUM_I}, {Dense});

  srand(53535);
  for (int i = 0; i < NUM_I; i++) {
    for (int j = 0; j < NUM_J; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      if (rand_float < SPARSITY) {
        A.insert({i, j}, (double) ((int) (rand_float*3/SPARSITY)));
      }
    }
  }

  for (int j = 0; j < NUM_J; j++) {
    float rand_float = (float)rand()/(float)(RAND_MAX);
    x.insert({j}, (double) ((int) (rand_float*3/SPARSITY)));
  }

  x.pack();
  A.pack();

  IndexVar i0("i0"), i1("i1"), kpos("kpos"), kpos0("kpos0"), kpos1("kpos1");
  y(i) = A(i, j) * x(j);

  IndexStmt stmt = y.getAssignment().concretize();
  stmt = stmt.fuse(i, j, k)
          .pos(k, kpos, A(i, j))
          .split(kpos, kpos0, kpos1, CHUNK_SIZE)
          .parallelize(kpos0, PARALLEL_UNIT::CPU_THREAD, OUTPUT_RACE_STRATEGY::ATOMICS);

  printToFile("example_spmv_cpu_splitpos", stmt);

  y.compile(stmt);
  y.assemble();
  y.compute();

  Tensor<double> expected("expected", {NUM_I}, {Dense});
  expected(i) = A(i, j) * x(j);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, y);
}

TEST(scheduling_eval, spmmCPU) {
  if (should_use_CUDA_codegen()) {
    return;
  }
  int NUM_I = 1021/10;
  int NUM_J = 1039/10;
  int NUM_K = 128;
  float SPARSITY = .3;
  Tensor<double> A("A", {NUM_I, NUM_J}, CSR);
  Tensor<double> B("B", {NUM_J, NUM_K}, {Dense, Dense});
  Tensor<double> C("C", {NUM_I, NUM_K}, {Dense, Dense});

  srand(75883);
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

  C(i, k) = A(i, j) * B(j, k);

  IndexStmt stmt = C.getAssignment().concretize();
  stmt = scheduleSpMMCPU(stmt, A);

  printToFile("spmm_cpu", stmt);

  C.compile(stmt);
  C.assemble();
  C.compute();

  Tensor<double> expected("expected", {NUM_I, NUM_K}, {Dense, Dense});
  expected(i, k) = A(i, j) * B(j, k);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, C);
}

TEST(scheduling_eval, sddmmCPU) {
  if (should_use_CUDA_codegen()) {
    return;
  }
  int NUM_I = 1021/10;
  int NUM_J = 1039/10;
  int NUM_K = 1057/10;
  float SPARSITY = .3;
  Tensor<double> A("A", {NUM_I, NUM_K}, {Dense, Dense});
  Tensor<double> B("B", {NUM_I, NUM_K}, CSR);
  Tensor<double> C("C", {NUM_I, NUM_J}, {Dense, Dense});
  Tensor<double> D("D", {NUM_J, NUM_K}, {Dense, Dense});

  srand(268238);
  for (int i = 0; i < NUM_I; i++) {
    for (int j = 0; j < NUM_J; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      C.insert({i, j}, (double) ((int) (rand_float*3/SPARSITY)));
    }
  }

  for (int i = 0; i < NUM_I; i++) {
    for (int k = 0; k < NUM_K; k++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      if (rand_float < SPARSITY) {
        B.insert({i, k}, (double) ((int) (rand_float*3/SPARSITY)));
      }
    }
  }

  for (int j = 0; j < NUM_J; j++) {
    for (int k = 0; k < NUM_K; k++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      D.insert({j, k}, (double) ((int) (rand_float*3/SPARSITY)));
    }
  }

  B.pack();
  C.pack();
  D.pack();

  A(i,k) = B(i,k) * C(i,j) * D(j,k);

  IndexStmt stmt = A.getAssignment().concretize();
  stmt = scheduleSDDMMCPU(stmt, B);

  printToFile("sddmm_cpu", stmt);

  A.compile(stmt);
  A.assemble();
  A.compute();

  Tensor<double> expected("expected", {NUM_I, NUM_K}, {Dense, Dense});
  expected(i,k) = B(i,k) * C(i,j) * D(j,k);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, A);
}

TEST(scheduling_eval, spmvCPU) {
  if (should_use_CUDA_codegen()) {
    return;
  }
  int NUM_I = 1021/10;
  int NUM_J = 1039/10;
  float SPARSITY = .3;
  Tensor<double> A("A", {NUM_I, NUM_J}, CSR);
  Tensor<double> x("x", {NUM_J}, {Dense});
  Tensor<double> y("y", {NUM_I}, {Dense});

  srand(120);
  for (int i = 0; i < NUM_I; i++) {
    for (int j = 0; j < NUM_J; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      if (rand_float < SPARSITY) {
        A.insert({i, j}, (double) ((int) (rand_float * 3 / SPARSITY)));
      }
    }
  }

  for (int j = 0; j < NUM_J; j++) {
    float rand_float = (float)rand()/(float)(RAND_MAX);
    x.insert({j}, (double) ((int) (rand_float*3/SPARSITY)));
  }

  x.pack();
  A.pack();

  y(i) = A(i, j) * x(j);

  IndexStmt stmt = y.getAssignment().concretize();
  stmt = scheduleSpMVCPU(stmt);

  printToFile("spmv_cpu", stmt);

  y.compile(stmt);
  y.assemble();
  y.compute();

  Tensor<double> expected("expected", {NUM_I}, {Dense});
  expected(i) = A(i, j) * x(j);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, y);
}

TEST(scheduling_eval, ttvCPU) {
  if (should_use_CUDA_codegen()) {
    return;
  }
  int NUM_I = 1021/10;
  int NUM_J = 1039/10;
  int NUM_K = 1057/10;
  float SPARSITY = .3;
  Tensor<double> A("A", {NUM_I, NUM_J}, {Dense, Dense}); // TODO: change to sparse outputs
  Tensor<double> B("B", {NUM_I, NUM_J, NUM_K}, {Sparse, Sparse, Sparse});
  Tensor<double> c("c", {NUM_K}, {Dense});

  srand(9536);
  for (int i = 0; i < NUM_I; i++) {
    for (int j = 0; j < NUM_J; j++) {
      for (int k = 0; k < NUM_K; k++) {
        float rand_float = (float) rand() / (float) (RAND_MAX);
        if (rand_float < SPARSITY) {
          B.insert({i, j, k}, (double) ((int) (rand_float * 3 / SPARSITY)));
        }
      }
    }
  }

  for (int k = 0; k < NUM_K; k++) {
    float rand_float = (float)rand()/(float)(RAND_MAX);
    c.insert({k}, (double) ((int) (rand_float*3)));
  }

  B.pack();
  c.pack();

  A(i,j) = B(i,j,k) * c(k);

  IndexStmt stmt = A.getAssignment().concretize();
  stmt = scheduleTTVCPU(stmt, B);

  printToFile("ttv_cpu", stmt);

  A.compile(stmt);
  A.assemble();
  A.compute();

  Tensor<double> expected("expected", {NUM_I, NUM_J}, {Dense, Dense});
  expected(i,j) = B(i,j,k) * c(k);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, A);
}

TEST(scheduling_eval, ttmCPU) {
  if (should_use_CUDA_codegen()) {
    return;
  }
  int NUM_I = 1021/40;
  int NUM_J = 1039/40;
  int NUM_K = 1057/40;
  int NUM_L = 1232/40;
  float SPARSITY = .1;
  Tensor<double> A("A", {NUM_I, NUM_J, NUM_L}, {Dense, Dense, Dense}); // TODO: change to sparse outputs
  Tensor<double> B("B", {NUM_I, NUM_J, NUM_K}, {Sparse, Sparse, Sparse});
  Tensor<double> C("C", {NUM_K, NUM_L}, {Dense, Dense});

  srand(935);
  for (int i = 0; i < NUM_I; i++) {
    for (int j = 0; j < NUM_J; j++) {
      for (int k = 0; k < NUM_K; k++) {
        float rand_float = (float) rand() / (float) (RAND_MAX);
        if (rand_float < SPARSITY) {
          B.insert({i, j, k}, (double) ((int) (rand_float * 3 / SPARSITY)));
        }
      }
    }
  }

  for (int k = 0; k < NUM_K; k++) {
    for (int l = 0; l < NUM_L; l++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      C.insert({k, l}, (double) ((int) (rand_float*3)));
    }
  }

  B.pack();
  C.pack();

  A(i,j,l) = B(i,j,k) * C(k,l);

  IndexStmt stmt = A.getAssignment().concretize();
  stmt = scheduleTTMCPU(stmt, B);

  printToFile("ttm_cpu", stmt);

  A.compile(stmt);
  A.assemble();
  A.compute();

  Tensor<double> expected("expected", {NUM_I, NUM_J, NUM_L}, {Dense, Dense, Dense});
  expected(i,j,l) = B(i,j,k) * C(k,l);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, A);
}

TEST(scheduling_eval, mttkrpCPU) {
  if (should_use_CUDA_codegen()) {
    return;
  }
  int NUM_I = 1021/20;
  int NUM_J = 1039/20;
  int NUM_K = 1057/20;
  int NUM_L = 1232/20;
  float SPARSITY = .1;
  Tensor<double> A("A", {NUM_I, NUM_J}, {Dense, Dense});
  Tensor<double> B("B", {NUM_I, NUM_K, NUM_L}, {Sparse, Sparse, Sparse});
  Tensor<double> C("C", {NUM_K, NUM_J}, {Dense, Dense});
  Tensor<double> D("D", {NUM_L, NUM_J}, {Dense, Dense});

  srand(549694);
  for (int i = 0; i < NUM_I; i++) {
    for (int k = 0; k < NUM_K; k++) {
      for (int l = 0; l < NUM_L; l++) {
        float rand_float = (float) rand() / (float) (RAND_MAX);
        if (rand_float < SPARSITY) {
          B.insert({i, k, l}, (double) ((int) (rand_float * 3 / SPARSITY)));
        }
      }
    }
  }

  for (int k = 0; k < NUM_K; k++) {
    for (int j = 0; j < NUM_J; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      C.insert({k, j}, (double) ((int) (rand_float*3)));
    }
  }

  for (int l = 0; l < NUM_L; l++) {
    for (int j = 0; j < NUM_J; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      D.insert({l, j}, (double) ((int) (rand_float*3)));
    }
  }

  B.pack();
  C.pack();
  D.pack();

  A(i,j) = B(i,k,l) * C(k,j) * D(l,j);

  IndexStmt stmt = A.getAssignment().concretize();
  stmt = scheduleMTTKRPCPU(stmt, B);
  printToFile("mttkrp_cpu", stmt);

  A.compile(stmt);
  A.assemble();
  A.compute();

  Tensor<double> expected("expected", {NUM_I, NUM_J}, {Dense, Dense});
  expected(i,j) = B(i,k,l) * C(k,j) * D(l,j);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, A);
}

TEST(scheduling_eval, spmvGPU) {
  if (!should_use_CUDA_codegen()) {
    return;
  }
  int NUM_I = 1021/10;
  int NUM_J = 1039/10;
  float SPARSITY = .01;
  Tensor<double> A("A", {NUM_I, NUM_J}, CSR);
  Tensor<double> x("x", {NUM_J}, {Dense});
  Tensor<double> y("y", {NUM_I}, {Dense});

  srand(94353);
  for (int i = 0; i < NUM_I; i++) {
    for (int j = 0; j < NUM_J; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      if (rand_float < SPARSITY) {
        A.insert({i, j}, (double) ((int) (rand_float * 3 / SPARSITY)));
      }
    }
  }

  for (int j = 0; j < NUM_J; j++) {
    float rand_float = (float)rand()/(float)(RAND_MAX);
    x.insert({j}, (double) ((int) (rand_float*3/SPARSITY)));
  }

  x.pack();
  A.pack();
  IndexExpr precomputed = A(i, j) * x(j);
  y(i) = precomputed;

  IndexStmt stmt = y.getAssignment().concretize();
  stmt = scheduleSpMVGPU(stmt, A, precomputed);

  printToFile("spmv_gpu", stmt);

  y.compile(stmt);
  y.assemble();
  y.compute();

  Tensor<double> expected("expected", {NUM_I}, {Dense});
  expected(i) = A(i, j) * x(j);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, y);
}

TEST(scheduling_eval, spmmGPU) {
  if (!should_use_CUDA_codegen()) {
    return;
  }
  int NUM_I = 1021/10;
  int NUM_J = 1039/10;
  int NUM_K = 32;
  float SPARSITY = .3;
  Tensor<double> A("A", {NUM_I, NUM_J}, CSR);
  Tensor<double> B("B", {NUM_J, NUM_K}, {Dense, Dense});
  Tensor<double> C("C", {NUM_I, NUM_K}, {Dense, Dense});

  srand(434321);
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
  IndexExpr precomputed = A(i, j) * B(j, k);
  C(i, k) = precomputed;

  IndexStmt stmt = C.getAssignment().concretize();
  stmt = scheduleSpMMGPU(stmt, A, precomputed);

  printToFile("spmm_gpu", stmt);

  C.compile(stmt);
  C.assemble();
  C.compute();

  Tensor<double> expected("expected", {NUM_I, NUM_K}, {Dense, Dense});
  expected(i, k) = A(i, j) * B(j, k);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, C);
}

TEST(scheduling_eval, spmmDCSRGPU) {
  if (!should_use_CUDA_codegen()) {
    return;
  }
  int NUM_I = 1021/10;
  int NUM_J = 1039/10;
  int NUM_K = 128;
  float SPARSITY = .3;
  Tensor<double> A("A", {NUM_I, NUM_J}, {Sparse, Sparse});
  Tensor<double> B("B", {NUM_J, NUM_K}, {Dense, Dense});
  Tensor<double> C("C", {NUM_I, NUM_K}, {Dense, Dense});

  srand(25643);
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

  C(i, k) = A(i, j) * B(j, k);

  IndexStmt stmt = C.getAssignment().concretize();
  stmt = scheduleSpMMNZRowsGPU(stmt, A);

  printToFile("spmm_dcsr_gpu", stmt);

  C.compile(stmt);
  C.assemble();
  C.compute();

  Tensor<double> expected("expected", {NUM_I, NUM_K}, {Dense, Dense});
  expected(i, k) = A(i, j) * B(j, k);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, C);
}

TEST(scheduling_eval, sddmmGPU) {
  if (!should_use_CUDA_codegen()) {
    return;
  }
  int NUM_I = 1021/10;
  int NUM_K = 1039/10;
  int NUM_J = 128;
  float SPARSITY = .3;
  Tensor<double> A("A", {NUM_I, NUM_K}, {Dense, Dense});
  Tensor<double> B("B", {NUM_I, NUM_K}, CSR);
  Tensor<double> C("C", {NUM_I, NUM_J}, {Dense, Dense});
  Tensor<double> D("D", {NUM_J, NUM_K}, {Dense, Dense});

  srand(535366);
  for (int i = 0; i < NUM_I; i++) {
    for (int j = 0; j < NUM_J; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      C.insert({i, j}, (double) ((int) (rand_float*3/SPARSITY)));
    }
  }

  for (int i = 0; i < NUM_I; i++) {
    for (int k = 0; k < NUM_K; k++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      if (rand_float < SPARSITY) {
        B.insert({i, k}, (double) ((int) (rand_float*3/SPARSITY)));
      }
    }
  }

  for (int j = 0; j < NUM_J; j++) {
    for (int k = 0; k < NUM_K; k++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      D.insert({j, k}, (double) ((int) (rand_float*3/SPARSITY)));
    }
  }

  B.pack();
  C.pack();
  D.pack();

  A(i,k) = B(i,k) * C(i,j) * D(j,k);

  IndexStmt stmt = A.getAssignment().concretize();
  stmt = scheduleSDDMMGPU(stmt, B);

  printToFile("sddmm_gpu", stmt);

  A.compile(stmt);
  A.assemble();
  A.compute();

  Tensor<double> expected("expected", {NUM_I, NUM_K}, {Dense, Dense});
  expected(i,k) = B(i,k) * C(i,j) * D(j,k);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, A);
}

TEST(scheduling_eval, ttmGPU) {
  if (!should_use_CUDA_codegen()) {
    return;
  }
  int NUM_I = 1021/40;
  int NUM_J = 1039/40;
  int NUM_K = 1232/40;
  int NUM_L = 128;
  float SPARSITY = .1;
  Tensor<double> A("A", {NUM_I, NUM_J, NUM_L}, {Dense, Dense, Dense}); // TODO: change to sparse outputs
  Tensor<double> B("B", {NUM_I, NUM_J, NUM_K}, {Sparse, Sparse, Sparse});
  Tensor<double> C("C", {NUM_K, NUM_L}, {Dense, Dense});

  srand(34644);
  for (int i = 0; i < NUM_I; i++) {
    for (int j = 0; j < NUM_J; j++) {
      for (int k = 0; k < NUM_K; k++) {
        float rand_float = (float) rand() / (float) (RAND_MAX);
        if (rand_float < SPARSITY) {
          B.insert({i, j, k}, (double) ((int) (rand_float * 3 / SPARSITY)));
        }
      }
    }
  }

  for (int k = 0; k < NUM_K; k++) {
    for (int l = 0; l < NUM_L; l++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      C.insert({k, l}, (double) ((int) (rand_float*3)));
    }
  }

  B.pack();
  C.pack();

  A(i,j,l) = B(i,j,k) * C(k,l);

  IndexStmt stmt = A.getAssignment().concretize();
  stmt = scheduleTTMGPU(stmt, B);

  printToFile("ttm_gpu", stmt);

  A.compile(stmt);
  A.assemble();
  A.compute();

  Tensor<double> expected("expected", {NUM_I, NUM_J, NUM_L}, {Dense, Dense, Dense});
  expected(i,j,l) = B(i,j,k) * C(k,l);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, A);
}

TEST(scheduling_eval, ttvGPU) {
  if (!should_use_CUDA_codegen()) {
    return;
  }
  int NUM_I = 1021/10;
  int NUM_J = 1039/10;
  int NUM_K = 1057/10;
  float SPARSITY = .3;
  Tensor<double> A("A", {NUM_I, NUM_J}, {Dense, Dense}); // TODO: change to sparse outputs
  Tensor<double> B("B", {NUM_I, NUM_J, NUM_K}, {Sparse, Sparse, Sparse});
  Tensor<double> c("c", {NUM_K}, {Dense});

  srand(35325);
  for (int i = 0; i < NUM_I; i++) {
    for (int j = 0; j < NUM_J; j++) {
      for (int k = 0; k < NUM_K; k++) {
        float rand_float = (float) rand() / (float) (RAND_MAX);
        if (rand_float < SPARSITY) {
          B.insert({i, j, k}, (double) ((int) (rand_float * 3 / SPARSITY)));
        }
      }
    }
  }

  for (int k = 0; k < NUM_K; k++) {
    float rand_float = (float)rand()/(float)(RAND_MAX);
    c.insert({k}, (double) ((int) (rand_float*3)));
  }

  B.pack();
  c.pack();

  IndexExpr precomputedExpr = B(i,j,k) * c(k);
  A(i,j) = precomputedExpr;

  IndexStmt stmt = A.getAssignment().concretize();
  stmt = scheduleTTVGPU(stmt, B, precomputedExpr);

  printToFile("ttv_gpu", stmt);

  A.compile(stmt);
  A.assemble();
  A.compute();

  Tensor<double> expected("expected", {NUM_I, NUM_J}, {Dense, Dense});
  expected(i,j) = B(i,j,k) * c(k);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, A);
}

TEST(scheduling_eval, mttkrpGPU) {
  if (!should_use_CUDA_codegen()) {
    return;
  }
  int NUM_I = 1021/40;
  int NUM_J = 128;
  int NUM_K = 1039/40;
  int NUM_L = 1232/40;
  float SPARSITY = .1;
  Tensor<double> A("A", {NUM_I, NUM_J}, {Dense, Dense});
  Tensor<double> B("B", {NUM_I, NUM_K, NUM_L}, {Sparse, Sparse, Sparse});
  Tensor<double> C("C", {NUM_K, NUM_J}, {Dense, Dense});
  Tensor<double> D("D", {NUM_L, NUM_J}, {Dense, Dense});

  srand(5464164);
  for (int i = 0; i < NUM_I; i++) {
    for (int k = 0; k < NUM_K; k++) {
      for (int l = 0; l < NUM_L; l++) {
        float rand_float = (float) rand() / (float) (RAND_MAX);
        if (rand_float < SPARSITY) {
          B.insert({i, k, l}, (double) ((int) (rand_float * 3 / SPARSITY)));
        }
      }
    }
  }

  for (int k = 0; k < NUM_K; k++) {
    for (int j = 0; j < NUM_J; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      C.insert({k, j}, (double) ((int) (rand_float*3)));
    }
  }

  for (int l = 0; l < NUM_L; l++) {
    for (int j = 0; j < NUM_J; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      D.insert({l, j}, (double) ((int) (rand_float*3)));
    }
  }

  B.pack();
  C.pack();
  D.pack();

  A(i,j) = B(i,k,l) * C(k,j) * D(l,j);

  IndexStmt stmt = A.getAssignment().concretize();
  stmt = scheduleMTTKRPGPU(stmt, B);

  printToFile("mttkrp_gpu", stmt);

  A.compile(stmt);
  A.assemble();
  A.compute();

  Tensor<double> expected("expected", {NUM_I, NUM_J}, {Dense, Dense});
  expected(i,j) = B(i,k,l) * C(k,j) * D(l,j);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, A);
}

TEST(generate_evaluation_files, cpu) {
  if (should_use_CUDA_codegen()) {
    return;
  }
  vector<vector<int>> spmv_parameters = {{8}, {16}, {32}};

  // 4 to 512 and 4, 8, 16
  vector<vector<int>> spmm_dcsr_parameters = {{16, 8}};
  vector<vector<int>> spmm_parameters = {};

  for (int i = 4; i <= 512; i *= 2) {
    for (int j = 4; j <= 16; j *= 2) {
      spmm_parameters.push_back({i,j});
    }
  }

  vector<vector<int>> mttkrp_parameters = spmm_parameters;
  vector<vector<int>> sddmm_parameters = {{16, 8}, {8, 8}};
  vector<vector<int>> ttv_parameters = {{16}, {8}, {32}};
  vector<vector<int>> ttm_parameters = {{16, 8}, {8, 8}};

  int NUM_I = 100;
  int NUM_J = 100;
  int NUM_K = 100;
  int NUM_L = 100;

  string file_ending = should_use_CUDA_codegen() ? ".cu" : ".c";
  string file_path = "eval_prepared_cpu/";
  mkdir(file_path.c_str(), 0777);

  // spmv
  {
    stringstream source;
    std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(source, ir::CodeGen::ImplementationGen);
    Tensor<double> A("A", {NUM_I, NUM_J}, CSR);
    Tensor<double> x("x", {NUM_J}, {Dense});
    Tensor<double> y("y", {NUM_I}, {Dense});
    y(i) = A(i, j) * x(j);
    IndexStmt stmt = y.getAssignment().concretize();
    bool isFirst = true;
    for (auto paramSet : spmv_parameters) {
      IndexStmt scheduled = scheduleSpMVCPU(stmt, paramSet[0]);
      ir::Stmt compute = lower(scheduled, string("compute_") + util::join(paramSet, "_"),  false, true);
      codegen->compile(compute, isFirst);
      isFirst = false;
    }
    ofstream source_file;
    source_file.open(file_path + "spmv_cpu" + file_ending);
    source_file << source.str();
    source_file.close();
  }

  // spmm
  {
    stringstream source;
    std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(source, ir::CodeGen::ImplementationGen);
    Tensor<double> A("A", {NUM_I, NUM_J}, CSR);
    Tensor<double> B("B", {NUM_J, NUM_K}, {Dense, Dense});
    Tensor<double> C("C", {NUM_I, NUM_K}, {Dense, Dense});
    C(i, k) = A(i, j) * B(j, k);
    IndexStmt stmt = C.getAssignment().concretize();
    bool isFirst = true;
    for (auto paramSet : spmm_parameters) {
      IndexStmt scheduled = scheduleSpMMCPU(stmt, A, paramSet[0], paramSet[1]);
      ir::Stmt compute = lower(scheduled, string("compute_") + util::join(paramSet, "_"),  false, true);
      codegen->compile(compute, isFirst);
      isFirst = false;
    }
    ofstream source_file;
    source_file.open(file_path + "spmm_cpu" + file_ending);
    source_file << source.str();
    source_file.close();
  }

  // sddmm
  {
    stringstream source;
    std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(source, ir::CodeGen::ImplementationGen);
    Tensor<double> A("A", {NUM_I, NUM_K}, {Dense, Dense});
    Tensor<double> B("B", {NUM_I, NUM_K}, CSR);
    Tensor<double> C("C", {NUM_I, NUM_J}, {Dense, Dense});
    Tensor<double> D("D", {NUM_J, NUM_K}, {Dense, Dense});
    A(i,k) = B(i,k) * C(i,j) * D(j,k);
    IndexStmt stmt = A.getAssignment().concretize();
    bool isFirst = true;
    for (auto paramSet : sddmm_parameters) {
      IndexStmt scheduled = scheduleSDDMMCPU(stmt, B, paramSet[0], paramSet[1]);
      ir::Stmt compute = lower(scheduled, string("compute_") + util::join(paramSet, "_"),  false, true);
      codegen->compile(compute, isFirst);
      isFirst = false;
    }
    ofstream source_file;
    source_file.open(file_path + "sddmm_cpu" + file_ending);
    source_file << source.str();
    source_file.close();
  }

  // ttv
  {
    stringstream source;
    std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(source, ir::CodeGen::ImplementationGen);
    Tensor<double> A("A", {NUM_I, NUM_J}, {Dense, Dense}); // TODO: change to sparse outputs
    Tensor<double> B("B", {NUM_I, NUM_J, NUM_K}, {Sparse, Sparse, Sparse});
    Tensor<double> c("c", {NUM_K}, {Dense});
    A(i,j) = B(i,j,k) * c(k);
    IndexStmt stmt = A.getAssignment().concretize();
    bool isFirst = true;
    for (auto paramSet : ttv_parameters) {
      IndexStmt scheduled = scheduleTTVCPU(stmt, B, paramSet[0]);
      ir::Stmt compute = lower(scheduled, string("compute_") + util::join(paramSet, "_"),  false, true);
      codegen->compile(compute, isFirst);
      isFirst = false;
    }
    ofstream source_file;
    source_file.open(file_path + "ttv_cpu" + file_ending);
    source_file << source.str();
    source_file.close();
  }

  // ttm
  {
    stringstream source;
    std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(source, ir::CodeGen::ImplementationGen);
    Tensor<double> A("A", {NUM_I, NUM_J, NUM_L}, {Dense, Dense, Dense}); // TODO: change to sparse outputs
    Tensor<double> B("B", {NUM_I, NUM_J, NUM_K}, {Sparse, Sparse, Sparse});
    Tensor<double> C("C", {NUM_K, NUM_L}, {Dense, Dense});
    A(i,j,l) = B(i,j,k) * C(k,l);
    IndexStmt stmt = A.getAssignment().concretize();
    bool isFirst = true;
    for (auto paramSet : ttm_parameters) {
      IndexStmt scheduled = scheduleTTMCPU(stmt, B, paramSet[0], paramSet[1]);
      ir::Stmt compute = lower(scheduled, string("compute_") + util::join(paramSet, "_"),  false, true);
      codegen->compile(compute, isFirst);
      isFirst = false;
    }
    ofstream source_file;
    source_file.open(file_path + "ttm_cpu" + file_ending);
    source_file << source.str();
    source_file.close();
  }

  // mttkrp
  {
    stringstream source;
    std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(source, ir::CodeGen::ImplementationGen);
    Tensor<double> A("A", {NUM_I, NUM_J}, {Dense, Dense});
    Tensor<double> B("B", {NUM_I, NUM_K, NUM_L}, {Sparse, Sparse, Sparse});
    Tensor<double> C("C", {NUM_K, NUM_J}, {Dense, Dense});
    Tensor<double> D("D", {NUM_L, NUM_J}, {Dense, Dense});
    A(i,j) = B(i,k,l) * C(k,j) * D(l,j);
    IndexStmt stmt = A.getAssignment().concretize();
    bool isFirst = true;
    for (auto paramSet : mttkrp_parameters) {
      IndexStmt scheduled = scheduleMTTKRPCPU(stmt, B, paramSet[0], paramSet[1]);
      ir::Stmt compute = lower(scheduled, string("compute_") + util::join(paramSet, "_"),  false, true);
      codegen->compile(compute, isFirst);
      isFirst = false;
    }
    ofstream source_file;
    source_file.open(file_path + "mttkrp_cpu" + file_ending);
    source_file << source.str();
    source_file.close();
  }
}

TEST(generate_evaluation_files, gpu) {
  if (!should_use_CUDA_codegen()) {
    return;
  }

  vector<vector<int>> spmv_parameters = {}; // {NNZ_PER_THREAD, BLOCK_SIZE}
  for (int i = 3; i <= 20; i++) {
    spmv_parameters.push_back({i, 512});
  }
  vector<vector<int>> spmm_parameters = {}; // {NNZ_PER_WARP, BLOCK_SIZE, CO_FACTOR}

  // 4, 8, ... 32 for NNZ_PER_WARP 512 block size
  for (int i = 4; i <= 32; i += 4) {
    spmm_parameters.push_back({i,512, 1});
  }

  vector<vector<int>> mttkrp_parameters = spmm_parameters; // {NNZ_PER_WARP, BLOCK_SIZE, CO_FACTOR}

  vector<vector<int>> spmm_dcsr_parameters = {{4, 256, 4}}; // {NNZ_PER_WARP, BLOCK_SIZE, CO_FACTOR}
  vector<vector<int>> sddmm_parameters = {{8*32, 256, 4}, {4*32, 512, 4}}; // {NNZ_PER_WARP, BLOCK_SIZE, CO_FACTOR}
  vector<vector<int>> ttv_parameters = {{8*32, 256}, {4*32, 512}}; // {NNZ_PER_WARP, BLOCK_SIZE}
  vector<vector<int>> ttm_parameters = {{8*32, 256, 4}, {4*32, 512, 8}}; // {NNZ_PER_WARP, BLOCK_SIZE, CO_FACTOR}

  int NUM_I = 100;
  int NUM_J = 100;
  int NUM_K = 100;
  int NUM_L = 100;

  string file_ending = should_use_CUDA_codegen() ? ".cu" : ".c";
  string file_path = "eval_prepared_gpu/";
  mkdir(file_path.c_str(), 0777);

  // spmv load-balance
  {
    stringstream source;
    std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(source, ir::CodeGen::ImplementationGen);
    Tensor<double> A("A", {NUM_I, NUM_J}, CSR);
    Tensor<double> x("x", {NUM_J}, {Dense});
    Tensor<double> y("y", {NUM_I}, {Dense});
    IndexExpr precomputed = A(i, j) * x(j);
    y(i) = precomputed;
    IndexStmt stmt = y.getAssignment().concretize();
    bool isFirst = true;

    IndexStmt scheduled = scheduleSpMVRowsGPU(stmt, A, precomputed);
    ir::Stmt compute = lower(scheduled, string("compute_rows"),  false, true);
    codegen->compile(compute, isFirst);
    isFirst = false;

    scheduled = scheduleSpMVSplitPosGPU(stmt, A, precomputed);
    compute = lower(scheduled, string("compute_nnz"),  false, true);
    codegen->compile(compute, isFirst);


    ofstream source_file;
    source_file.open(file_path + "spmv_gpu_load_balance" + file_ending);
    source_file << source.str();
    source_file.close();
  }

  // spmv
  {
    stringstream source;
    std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(source, ir::CodeGen::ImplementationGen);
    Tensor<double> A("A", {NUM_I, NUM_J}, CSR);
    Tensor<double> x("x", {NUM_J}, {Dense});
    Tensor<double> y("y", {NUM_I}, {Dense});
    IndexExpr precomputed = A(i, j) * x(j);
    y(i) = precomputed;
    IndexStmt stmt = y.getAssignment().concretize();
    bool isFirst = true;
    for (auto paramSet : spmv_parameters) {
      IndexStmt scheduled = scheduleSpMVGPU(stmt, A, precomputed, paramSet[0], paramSet[1]);
      ir::Stmt compute = lower(scheduled, string("compute_") + util::join(paramSet, "_"),  false, true);
      codegen->compile(compute, isFirst);
      isFirst = false;
    }
    ofstream source_file;
    source_file.open(file_path + "spmv_gpu" + file_ending);
    source_file << source.str();
    source_file.close();
  }

  // spmm
  {
    stringstream source;
    std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(source, ir::CodeGen::ImplementationGen);
    Tensor<double> A("A", {NUM_I, NUM_J}, CSR);
    bool isFirst = true;
    for (auto paramSet : spmm_parameters) {
      int NUM_K = paramSet[2] * WARP_SIZE;
      Tensor<double> B("B", {NUM_J, NUM_K}, {Dense, Dense});
      Tensor<double> C("C", {NUM_I, NUM_K}, {Dense, Dense});
      IndexExpr precomputed = A(i, j) * B(j, k);
      C(i, k) = precomputed;
      IndexStmt stmt = C.getAssignment().concretize();
      IndexStmt scheduled = scheduleSpMMGPU(stmt, A, precomputed, paramSet[0], paramSet[1], paramSet[2]);
      ir::Stmt compute = lower(scheduled, string("compute_") + util::join(paramSet, "_"),  false, true);
      codegen->compile(compute, isFirst);
      isFirst = false;
    }
    ofstream source_file;
    source_file.open(file_path + "spmm_gpu" + file_ending);
    source_file << source.str();
    source_file.close();
  }

  // sddmm
  {
    stringstream source;

    Tensor<double> C("C", {NUM_I, NUM_J}, {Dense, Dense});
    bool isFirst = true;
    for (auto paramSet : sddmm_parameters) {
      int NUM_K = paramSet[2] * WARP_SIZE;
      std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(source, ir::CodeGen::ImplementationGen);
      Tensor<double> A("A", {NUM_I, NUM_K}, {Dense, Dense});
      Tensor<double> B("B", {NUM_I, NUM_K}, CSR);
      Tensor<double> D("D", {NUM_J, NUM_K}, {Dense, Dense});
      A(i,k) = B(i,k) * C(i,j) * D(j,k);
      IndexStmt stmt = A.getAssignment().concretize();
      IndexStmt scheduled = scheduleSDDMMGPU(stmt, B, paramSet[0], paramSet[1], paramSet[2]);
      ir::Stmt compute = lower(scheduled, string("compute_") + util::join(paramSet, "_"),  false, true);
      codegen->compile(compute, isFirst);
      isFirst = false;
    }
    ofstream source_file;
    source_file.open(file_path + "sddmm_gpu" + file_ending);
    source_file << source.str();
    source_file.close();
  }

  // ttv
  {
    stringstream source;
    std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(source, ir::CodeGen::ImplementationGen);
    Tensor<double> A("A", {NUM_I, NUM_J}, {Dense, Dense}); // TODO: change to sparse outputs
    Tensor<double> B("B", {NUM_I, NUM_J, NUM_K}, {Sparse, Sparse, Sparse});
    Tensor<double> c("c", {NUM_K}, {Dense});
    IndexExpr precomputedExpr = B(i,j,k) * c(k);
    A(i,j) = precomputedExpr;
    IndexStmt stmt = A.getAssignment().concretize();
    bool isFirst = true;
    for (auto paramSet : ttv_parameters) {
      IndexStmt scheduled = scheduleTTVGPU(stmt, B, precomputedExpr, paramSet[0], paramSet[1]);
      ir::Stmt compute = lower(scheduled, string("compute_") + util::join(paramSet, "_"),  false, true);
      codegen->compile(compute, isFirst);
      isFirst = false;
    }
    ofstream source_file;
    source_file.open(file_path + "ttv_gpu" + file_ending);
    source_file << source.str();
    source_file.close();
  }

  // ttm
  {
    stringstream source;
    std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(source, ir::CodeGen::ImplementationGen);
    Tensor<double> A("A", {NUM_I, NUM_J, NUM_L}, {Dense, Dense, Dense}); // TODO: change to sparse outputs
    bool isFirst = true;
    for (auto paramSet : ttm_parameters) {
      int NUM_K = paramSet[2] * WARP_SIZE;
      Tensor<double> B("B", {NUM_I, NUM_J, NUM_K}, {Sparse, Sparse, Sparse});
      Tensor<double> C("C", {NUM_K, NUM_L}, {Dense, Dense});
      A(i,j,l) = B(i,j,k) * C(k,l);
      IndexStmt stmt = A.getAssignment().concretize();
      IndexStmt scheduled = scheduleTTMGPU(stmt, B, paramSet[0], paramSet[1], paramSet[2]);
      ir::Stmt compute = lower(scheduled, string("compute_") + util::join(paramSet, "_"),  false, true);
      codegen->compile(compute, isFirst);
      isFirst = false;
    }
    ofstream source_file;
    source_file.open(file_path + "ttm_gpu" + file_ending);
    source_file << source.str();
    source_file.close();
  }

  // mttkrp
  {
    stringstream source;
    std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(source, ir::CodeGen::ImplementationGen);
    Tensor<double> B("B", {NUM_I, NUM_K, NUM_L}, {Sparse, Sparse, Sparse});

    bool isFirst = true;
    for (auto paramSet : mttkrp_parameters) {
      int NUM_J = paramSet[2] * WARP_SIZE;
      Tensor<double> A("A", {NUM_I, NUM_J}, {Dense, Dense});
      Tensor<double> C("C", {NUM_K, NUM_J}, {Dense, Dense});
      Tensor<double> D("D", {NUM_L, NUM_J}, {Dense, Dense});
      A(i,j) = B(i,k,l) * C(k,j) * D(l,j);
      IndexStmt stmt = A.getAssignment().concretize();
      IndexStmt scheduled = scheduleMTTKRPGPU(stmt, B, paramSet[0], paramSet[1], paramSet[2]);
      ir::Stmt compute = lower(scheduled, string("compute_") + util::join(paramSet, "_"),  false, true);
      codegen->compile(compute, isFirst);
      isFirst = false;
    }
    ofstream source_file;
    source_file.open(file_path + "mttkrp_gpu" + file_ending);
    source_file << source.str();
    source_file.close();
  }
}

TEST(generate_figures, cpu) {
  if (should_use_CUDA_codegen()) {
    return;
  }

  int NUM_I = 100;
  int NUM_J = 100;
  int NUM_K = 100;
  int NUM_L = 100;

  string file_ending = should_use_CUDA_codegen() ? ".cu" : ".c";
  string file_path = "figures_cpu/";
  mkdir(file_path.c_str(), 0777);

  // spmv
  {
    stringstream source;
    std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(source, ir::CodeGen::ImplementationGen);
    Tensor<double> A("A", {NUM_I, NUM_J}, CSR);
    Tensor<double> x("x", {NUM_J}, {Dense});
    Tensor<double> y("y", {NUM_I}, {Dense});
    y(i) = A(i, j) * x(j);
    IndexStmt stmt = y.getAssignment().concretize();
    bool isFirst = true;
    string functionNames[] = {"spmv_unscheduled", "spmv_row_tiled", "spmv_pos_iteration"};
    IndexStmt  (* schedulingFunctions [])(IndexStmt, Tensor<double>) = {&exampleScheduleSPMVUntiled, &exampleScheduleSPMVCPURowTiling, &exampleScheduleSPMVPosIteration};

    int ii = 0;
    for (auto schedulingFunction : schedulingFunctions) {
      IndexStmt scheduled = schedulingFunction(stmt, A);
      ir::Stmt compute = lower(scheduled, functionNames[ii], false, true);
      codegen->compile(compute, isFirst);
      isFirst = false;
      ii++;
    }
    ofstream source_file;
    source_file.open(file_path + "fig_spmv" + file_ending);
    source_file << source.str();
    source_file.close();
  }
}