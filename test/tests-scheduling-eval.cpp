#include <taco/index_notation/transformations.h>
#include <codegen/codegen_c.h>
#include <codegen/codegen_cuda.h>
#include <fstream>
#include "test.h"
#include "test_tensors.h"
#include "taco/tensor.h"
#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/transformations.h"
#include "codegen/codegen.h"
#include "taco/lower/lower.h"
#include "op_factory.h"

using namespace taco;
const IndexVar i("i"), j("j"), k("k"), l("l"), m("m"), n("n");
int WARP_SIZE = 32;

void printToCout(IndexStmt stmt) {
  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);
  ir::Stmt compute = lower(stmt, "compute", false, true);
  codegen->compile(compute, true);
}

void printToFile(string filename, IndexStmt stmt) {
  stringstream source;

  string file_path = "eval_generated/";
  mkdir(file_path.c_str(), 0777);

  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(source, ir::CodeGen::ImplementationGen);
  ir::Stmt compute = lower(stmt, "compute",  true, true);
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
          .parallelize(i0, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces);
}

IndexStmt schedulePrecompute3D(IndexStmt stmt, IndexExpr precomputedExpr) {
  TensorVar precomputed("precomputed", Type(Float64, {16, 16, 16}), {Dense, Dense, Dense});
  return stmt.precompute(precomputedExpr, {i, j, l} , {i, j, l}, precomputed);
}

IndexStmt schedulePrecompute1D(IndexStmt stmt, IndexExpr precomputedExpr) {
  TensorVar precomputed("precomputed", Type(Float64, {102}), {Dense});
  return stmt.precompute(precomputedExpr, i , i, precomputed);
}

IndexStmt scheduleSpMMCPU(IndexStmt stmt, Tensor<double> A, int CHUNK_SIZE=16, int UNROLL_FACTOR=8) {
  IndexVar i0("i0"), i1("i1"), kbounded("kbounded"), k0("k0"), k1("k1"), jpos("jpos"), jpos0("jpos0"), jpos1("jpos1");
  return stmt.split(i, i0, i1, CHUNK_SIZE)
          .pos(j, jpos, A(i,j))
          .split(jpos, jpos0, jpos1, UNROLL_FACTOR)
          .reorder({i0, i1, jpos0, k, jpos1})
          .parallelize(i0, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces)
          .parallelize(k, ParallelUnit::CPUVector, OutputRaceStrategy::IgnoreRaces);
}

IndexStmt scheduleSpGEMMCPU(IndexStmt stmt, bool doPrecompute) {
  Assignment assign = stmt.as<Forall>().getStmt().as<Forall>().getStmt()
                          .as<Forall>().getStmt().as<Assignment>();
  TensorVar result = assign.getLhs().getTensorVar();

  stmt = reorderLoopsTopologically(stmt);
  if (doPrecompute) {
    IndexVar j = assign.getLhs().getIndexVars()[1];
    TensorVar w("w", Type(result.getType().getDataType(), 
                {result.getType().getShape().getDimension(1)}), taco::dense);
    stmt = stmt.precompute(assign.getRhs(), j, j, w);
  }
  stmt = stmt.assemble(result, AssembleStrategy::Insert, true);
  auto qi_stmt = stmt.as<Assemble>().getQueries();
  IndexVar qi;
  if (isa<Where>(qi_stmt)) {
    qi = qi_stmt.as<Where>().getConsumer().as<Forall>().getIndexVar();
  } else {
    qi = qi_stmt.as<Forall>().getIndexVar();
  }
  stmt = stmt.parallelize(i, ParallelUnit::CPUThread,
                          OutputRaceStrategy::NoRaces)
             .parallelize(qi, ParallelUnit::CPUThread,
                          OutputRaceStrategy::NoRaces);

  return stmt;
}

IndexStmt scheduleSpAddCPU(IndexStmt stmt) {
  IndexStmt body = stmt.as<Forall>().getStmt().as<Forall>().getStmt();
  if (isa<Forall>(body)) {
    body = body.as<Forall>().getStmt();
  }
  Assignment assign = body.as<Assignment>();
  TensorVar result = assign.getLhs().getTensorVar();

  stmt = reorderLoopsTopologically(stmt);
  stmt = stmt.assemble(result, AssembleStrategy::Insert, true);

  IndexVar qi = stmt.as<Assemble>().getQueries().as<Forall>().getIndexVar();
  stmt = stmt.parallelize(i, ParallelUnit::CPUThread,
                          OutputRaceStrategy::NoRaces)
             .parallelize(qi, ParallelUnit::CPUThread,
                          OutputRaceStrategy::NoRaces);

  return stmt;
}

IndexStmt scheduleSDDMMCPU(IndexStmt stmt, Tensor<double> B, int CHUNK_SIZE=16, int UNROLL_FACTOR=8) {
  IndexVar i0("i0"), i1("i1"), kpos("kpos"), kpos0("kpos0"), kpos1("kpos1");
  return stmt.split(i, i0, i1, CHUNK_SIZE)
          .pos(k, kpos, B(i,k))
          .split(kpos, kpos0, kpos1, UNROLL_FACTOR)
          .reorder({i0, i1, kpos0, j, kpos1})
          .parallelize(i0, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces)
          .parallelize(kpos1, ParallelUnit::CPUVector, OutputRaceStrategy::ParallelReduction);
}

IndexStmt scheduleTTVCPU(IndexStmt stmt, Tensor<double> B, int CHUNK_SIZE=16) {
  IndexVar f("f"), fpos("fpos"), chunk("chunk"), fpos2("fpos2");
  return stmt.fuse(i, j, f)
          .pos(f, fpos, B(i,j,k))
          .split(fpos, chunk, fpos2, CHUNK_SIZE)
          .reorder({chunk, fpos2, k})
          .parallelize(chunk, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces);
}

IndexStmt scheduleTTVCPUCSR(IndexStmt stmt) {
  TensorVar result = stmt.as<Forall>().getStmt().as<Forall>().getStmt()
                         .as<Forall>().getStmt().as<Assignment>().getLhs()
                         .getTensorVar();
  return stmt.assemble(result, AssembleStrategy::Insert)
             .parallelize(i, ParallelUnit::CPUThread, 
                          OutputRaceStrategy::NoRaces);
}

IndexStmt scheduleTTMCPU(IndexStmt stmt, Tensor<double> B, int CHUNK_SIZE=16, int UNROLL_FACTOR=8) {
  IndexVar f("f"), fpos("fpos"), chunk("chunk"), fpos2("fpos2"), kpos("kpos"), kpos1("kpos1"), kpos2("kpos2");
  return stmt.fuse(i, j, f)
          .pos(f, fpos, B(i,j,k))
          .split(fpos, chunk, fpos2, CHUNK_SIZE)
          .pos(k, kpos, B(i,j,k))
          .split(kpos, kpos1, kpos2, UNROLL_FACTOR)
          .reorder({chunk, fpos2, kpos1, l, kpos2})
          .parallelize(chunk, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces)
          .parallelize(kpos2, ParallelUnit::CPUVector, OutputRaceStrategy::ParallelReduction);;
}

IndexStmt scheduleMTTKRPCPU(IndexStmt stmt, Tensor<double> B, int CHUNK_SIZE=16, int UNROLL_FACTOR=8) {
  int NUM_J = 1039/20;
  IndexVar i1("i1"), i2("i2");

  IndexExpr precomputeExpr = stmt.as<Forall>().getStmt().as<Forall>().getStmt()
                                 .as<Forall>().getStmt().as<Forall>().getStmt()
                                 .as<Assignment>().getRhs().as<Mul>().getA();
  TensorVar w("w", Type(Float64, {(size_t)NUM_J}), taco::dense);

  stmt = stmt.split(i, i1, i2, CHUNK_SIZE)
    .reorder({i1, i2, k, l, j});
  stmt = stmt.precompute(precomputeExpr, j, j, w);

  return stmt
          .parallelize(i1, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces);
}

IndexStmt scheduleMTTKRPPrecomputedCPU(IndexStmt stmt, Tensor<double> B, int CHUNK_SIZE=16, int UNROLL_FACTOR=8) {
  IndexVar i1("i1"), i2("i2"), j_pre("j_pre");
  return stmt.split(i, i1, i2, CHUNK_SIZE)
          .parallelize(i1, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces);
}

IndexStmt scheduleMTTKRP4CPU(IndexStmt stmt, Tensor<double> B, int CHUNK_SIZE=16, int UNROLL_FACTOR=8) {
  IndexVar i1("i1"), i2("i2");
  return stmt.split(i, i1, i2, CHUNK_SIZE)
          .reorder({i1, i2, k, l, m, j})
          .parallelize(i1, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces);
}

IndexStmt scheduleMTTKRP5CPU(IndexStmt stmt, Tensor<double> B, int CHUNK_SIZE=16, int UNROLL_FACTOR=8) {
  IndexVar i1("i1"), i2("i2");
  return stmt.split(i, i1, i2, CHUNK_SIZE)
          .reorder({i1, i2, k, l, m, n, j})
          .parallelize(i1, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces);
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
          .parallelize(block, ParallelUnit::GPUBlock, OutputRaceStrategy::IgnoreRaces)
          .parallelize(warp, ParallelUnit::GPUWarp, OutputRaceStrategy::IgnoreRaces)
          .parallelize(thread, ParallelUnit::GPUThread, OutputRaceStrategy::Atomics);
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
          .parallelize(block, ParallelUnit::GPUBlock, OutputRaceStrategy::IgnoreRaces)
          .parallelize(warp, ParallelUnit::GPUWarp, OutputRaceStrategy::IgnoreRaces)
          .parallelize(thread, ParallelUnit::GPUThread, OutputRaceStrategy::Temporary);
}

IndexStmt scheduleSpMVThreadPerRowGPU(IndexStmt stmt, Tensor<double> A, IndexExpr precomputedExpr, int BLOCK_SIZE=256) {
  int ROWS_PER_TB = BLOCK_SIZE;
  IndexVar block("block"), warp("warp"), thread("thread"), thread_nz("thread_nz"), i1("i1"), jpos("jpos"), block_row("block_row"), warp_row("warp_row");
  return stmt.split(i, block, thread, ROWS_PER_TB)
          .parallelize(block, ParallelUnit::GPUBlock, OutputRaceStrategy::NoRaces)
          .parallelize(thread, ParallelUnit::GPUThread, OutputRaceStrategy::NoRaces);
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
          .parallelize(block, ParallelUnit::GPUBlock, OutputRaceStrategy::IgnoreRaces)
          .parallelize(warp, ParallelUnit::GPUWarp, OutputRaceStrategy::IgnoreRaces)
          .parallelize(thread, ParallelUnit::GPUThread, OutputRaceStrategy::Atomics);
}

IndexStmt scheduleSpMMGPU(IndexStmt stmt, Tensor<double> A, IndexExpr precomputedExpr, int NNZ_PER_WARP=8, int BLOCK_SIZE=256) {
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
          .reorder({block, warp, thread, dense_val_unbounded, nnz})
          //.precompute(precomputedExpr, nnz, nnz, precomputed)
          .bound(dense_val_unbounded, dense_val, 4, BoundType::MaxExact)
          //.unroll(dense_val, 4)
          .parallelize(block, ParallelUnit::GPUBlock, OutputRaceStrategy::IgnoreRaces)
          .parallelize(warp, ParallelUnit::GPUWarp, OutputRaceStrategy::IgnoreRaces)
          .parallelize(thread, ParallelUnit::GPUThread, OutputRaceStrategy::Atomics);
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
          .bound(dense_val_unbounded, dense_val, CO_FACTOR, BoundType::MaxExact)
          .reorder({block, warp, nnz, thread, dense_val})
          .unroll(dense_val, CO_FACTOR)
          .parallelize(block, ParallelUnit::GPUBlock, OutputRaceStrategy::IgnoreRaces)
          .parallelize(warp, ParallelUnit::GPUWarp, OutputRaceStrategy::Atomics)
          .parallelize(thread, ParallelUnit::GPUThread, OutputRaceStrategy::ParallelReduction);
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
          .bound(dense_val_unbounded, dense_val, CO_FACTOR, BoundType::MaxExact)
          .reorder({block, warp, nnz, thread, dense_val})
          .unroll(dense_val, CO_FACTOR)
          .parallelize(block, ParallelUnit::GPUBlock, OutputRaceStrategy::IgnoreRaces)
          .parallelize(warp, ParallelUnit::GPUWarp, OutputRaceStrategy::IgnoreRaces)
          .parallelize(thread, ParallelUnit::GPUThread, OutputRaceStrategy::Atomics);
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
          .parallelize(block, ParallelUnit::GPUBlock, OutputRaceStrategy::IgnoreRaces)
          .parallelize(warp, ParallelUnit::GPUWarp, OutputRaceStrategy::IgnoreRaces)
          .parallelize(thread, ParallelUnit::GPUThread, OutputRaceStrategy::Atomics);
}

IndexStmt scheduleMTTKRPGPU(IndexStmt stmt, Tensor<double> B, int NNZ_PER_WARP=16, int BLOCK_SIZE=256) {
  int NNZ_PER_TB = NNZ_PER_WARP * (BLOCK_SIZE / WARP_SIZE);
  IndexVar kl("kl"), f("f"), fpos("fpos"), block("block"), fpos1("fpos1"), warp("warp"), nnz("nnz"), dense_val_unbounded("dense_val_unbounded"), dense_val("dense_val"), thread("thread");
  return stmt.reorder({i,k,l,j})
          .fuse(k, l, kl)
          .fuse(i, kl, f)
          .pos(f, fpos, B(i, k, l))
          .split(fpos, block, fpos1, NNZ_PER_TB)
          .split(fpos1, warp, nnz, NNZ_PER_WARP)
          .split(j, dense_val_unbounded, thread, WARP_SIZE)
          .bound(dense_val_unbounded, dense_val, 1, BoundType::MaxExact)
          .reorder({block, warp, dense_val, thread, nnz})
          .parallelize(block, ParallelUnit::GPUBlock, OutputRaceStrategy::IgnoreRaces)
          .parallelize(warp, ParallelUnit::GPUWarp, OutputRaceStrategy::IgnoreRaces)
          .parallelize(thread, ParallelUnit::GPUThread, OutputRaceStrategy::Atomics);
}

// splits so same number of rows per warp and then each thread in warp gets 1/32 of the columns space
IndexStmt scheduleSpMMRowsGPU(IndexStmt stmt, Tensor<double> A, int ROWS_PER_WARP=4, int BLOCK_SIZE=256) {
  int ROWS_PER_TB = ROWS_PER_WARP * (BLOCK_SIZE / WARP_SIZE);
  IndexVar i1("i1"), block("block"), warp("warp"), warp_row("warp_row"), thread("thread"), thread_col("thread_col");
  return stmt.split(i, block, i1, ROWS_PER_TB)
          .split(i1, warp, warp_row, ROWS_PER_WARP)
          .split(k, thread, thread_col, 32)
          .reorder({block, warp, warp_row, thread, thread_col, j})
          .parallelize(block, ParallelUnit::GPUBlock, OutputRaceStrategy::IgnoreRaces)
          .parallelize(warp, ParallelUnit::GPUWarp, OutputRaceStrategy::IgnoreRaces)
          .parallelize(thread, ParallelUnit::GPUThread, OutputRaceStrategy::Atomics);
}

// splits so same number of nonzero rows per warp and then each thread in warp gets 1/32 of the columns space (no search needed)
IndexStmt scheduleSpMMNZRowsGPU(IndexStmt stmt, Tensor<double> A, int NZ_ROWS_PER_WARP=4, int BLOCK_SIZE=256) {
  int NZ_ROWS_PER_TB = NZ_ROWS_PER_WARP * (BLOCK_SIZE / WARP_SIZE);
  IndexVar ip("ip"), ip1("ip1"), block("block"), warp("warp"), warp_row("warp_row"), thread("thread"), thread_col("thread_col");
  return stmt.pos(i, ip, A(i, j))
          .split(ip, block, ip1, NZ_ROWS_PER_TB)
          .split(ip1, warp, warp_row, NZ_ROWS_PER_WARP)
          .split(k, thread, thread_col, 32)
          .reorder({block, warp, warp_row, thread, thread_col, j})
          .parallelize(block, ParallelUnit::GPUBlock, OutputRaceStrategy::IgnoreRaces)
          .parallelize(warp, ParallelUnit::GPUWarp, OutputRaceStrategy::IgnoreRaces)
          .parallelize(thread, ParallelUnit::GPUThread, OutputRaceStrategy::Atomics);
}


IndexStmt scheduleSpMMCPUNoVec(IndexStmt stmt, Tensor<double> A, int CHUNK_SIZE=16, int UNROLL_FACTOR=8) {
  IndexVar i0("i0"), i1("i1"), kbounded("kbounded"), k0("k0"), k1("k1"), jpos("jpos"), jpos0("jpos0"), jpos1("jpos1");
  return stmt.split(i, i0, i1, CHUNK_SIZE)
          .pos(j, jpos, A(i,j))
          .split(jpos, jpos0, jpos1, UNROLL_FACTOR)
          .reorder({i0, i1, jpos0, k, jpos1})
          .parallelize(i0, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces);
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
  Tensor<double> x("x", {NUM_J}, Format({Dense}));
  Tensor<double> y("y", {NUM_I}, Format({Dense}));

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


  y(i) = A(i, j) * x(j);
  IndexStmt stmt = y.getAssignment().concretize();
  stmt = stmt.parallelize(i, ParallelUnit::CPUThread, OutputRaceStrategy::Atomics);

  //printToFile("test_spmvCPU_temp", stmt);

  y.compile(stmt);
  y.assemble();
  y.compute();

  Tensor<double> expected("expected", {NUM_I}, Format({Dense}));
  expected(i) = A(i, j) * x(j);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, y);
}

TEST(scheduling_eval, test_sptvCPU_temp) {
  if (should_use_CUDA_codegen()) {
    return;
  }
  int NUM_I = 1021/10;
  int NUM_J = 1039/10;
  int NUM_K = 1049/10;
  float SPARSITY = .01;
  Tensor<double> A("A", {NUM_I, NUM_J, NUM_K}, Format({Sparse, Sparse, Sparse}));
  Tensor<double> x("x", {NUM_K}, Format({Dense}));
  Tensor<double> y("y", {NUM_J}, Format({Dense}));

  srand(4357);
  for (int i = 0; i < NUM_I; i++) {
    for (int j = 0; j < NUM_J; j++) {
      for (int k = 0; k < NUM_K; k++) {
        float rand_float = (float)rand()/(float)(RAND_MAX);
        if (rand_float < SPARSITY) {
          A.insert({i, j, k}, (double) ((int) (rand_float*3/SPARSITY)));
        }
      }
    }
  }

  for (int k = 0; k < NUM_K; k++) {
    float rand_float = (float)rand()/(float)(RAND_MAX);
    x.insert({k}, (double) ((int) (rand_float*3/SPARSITY)));
  }

  x.pack();
  A.pack();


  y(j) = A(i, j, k) * x(k);
  IndexStmt stmt = y.getAssignment().concretize();
  stmt = stmt.reorder({i,j,k}).parallelize(j, ParallelUnit::CPUThread, OutputRaceStrategy::Atomics);

  //printToFile("test_sptvCPU_temp", stmt);

  y.compile(stmt);
  y.assemble();
  y.compute();

  Tensor<double> expected("expected", {NUM_J}, Format({Dense}));
  expected(j) = A(i, j, k) * x(k);
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
  Tensor<double> x("x", {NUM_J}, Format({Dense}));
  Tensor<double> y("y", {NUM_I}, Format({Dense}));

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
          .parallelize(kpos0, ParallelUnit::CPUThread, OutputRaceStrategy::Atomics);

  //printToFile("example_spmv_cpu_splitpos", stmt);

  y.compile(stmt);
  y.assemble();
  y.compute();

  Tensor<double> expected("expected", {NUM_I}, Format({Dense}));
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

  //printToFile("spmm_cpu", stmt);

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

struct spgemm : public TestWithParam<std::tuple<Format,Format,bool>> {};

TEST_P(spgemm, scheduling_eval) {
  if (should_use_CUDA_codegen()) {
    return;
  }

  Format aFormat, bFormat;
  bool doPrecompute;
  std::tie(aFormat, bFormat, doPrecompute) = GetParam();

  int NUM_I = 100;
  int NUM_J = 100;
  int NUM_K = 100;
  float SPARSITY = .03;
  Tensor<double> A("A", {NUM_I, NUM_J}, aFormat);
  Tensor<double> B("B", {NUM_J, NUM_K}, bFormat);
  Tensor<double> C("C", {NUM_I, NUM_K}, CSR);

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
      if (rand_float < SPARSITY) {
        B.insert({j, k}, (double) ((int) (rand_float*3/SPARSITY)));
      }
    }
  }

  A.pack();
  B.pack();

  C(i, k) = A(i, j) * B(j, k);
  IndexStmt stmt = C.getAssignment().concretize();
  stmt = scheduleSpGEMMCPU(stmt, doPrecompute);

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

INSTANTIATE_TEST_CASE_P(spgemm, spgemm,
                        Values(std::make_tuple(CSR, CSR, true),
                               std::make_tuple(DCSR, CSR, true),
                               std::make_tuple(DCSR, DCSR, true),
                               std::make_tuple(CSR, CSC, true),
                               std::make_tuple(DCSR, DCSC, true)));

TEST(scheduling_eval, spmataddCPU) {
  if (should_use_CUDA_codegen()) {
    return;
  }

  int NUM_I = 1000;
  int NUM_J = 10;
  float SPARSITY = .15;
  Tensor<double> A("A", {NUM_I, NUM_J}, CSR);
  Tensor<double> B("B", {NUM_I, NUM_J}, CSR);
  Tensor<double> C("C", {NUM_I, NUM_J}, CSR);
  Tensor<double> eA("eA", {NUM_I, NUM_J}, Dense);
  Tensor<double> eB("eB", {NUM_I, NUM_J}, Dense);
  Tensor<double> eC("eC", {NUM_I, NUM_J}, Dense);

  srand(75883);
  for (int i = 0; i < NUM_I; i++) {
    for (int j = 0; j < NUM_J; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      if (rand_float < SPARSITY) {
        double val = (double)((int)(rand_float*3/SPARSITY));
        A.insert({i, j}, val);
        eA.insert({i, j}, val);
      }
    }
  }

  for (int i = 0; i < NUM_I; i++) {
    for (int j = 0; j < NUM_J; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      if (rand_float < SPARSITY) {
        double val = (double)((int)(rand_float*3/SPARSITY));
        B.insert({i, j}, val);
        eB.insert({i, j}, val);
      }
    }
  }

  A.pack();
  B.pack();

  C(i, j) = A(i, j) + B(i, j);
  IndexStmt stmt = C.getAssignment().concretize();
  stmt = scheduleSpAddCPU(stmt);

  C.compile(stmt);
  C.assemble();
  C.compute();

  eC(i, j) = eA(i, j) + eB(i, j);
  eC.compile();
  eC.assemble();
  eC.compute();
  ASSERT_TENSOR_EQ(eC, C);
}

TEST(scheduling_eval, sptenaddCPU) {
  if (should_use_CUDA_codegen()) {
    return;
  }

  int NUM_I = 100;
  int NUM_J = 10;
  int NUM_K = 10;
  float SPARSITY = .02;
  Format ecsr({Dense, Compressed(ModeFormat::NOT_UNIQUE), 
               Singleton(ModeFormat::UNIQUE)});
  Tensor<double> A("A", {NUM_I, NUM_J, NUM_K}, ecsr);
  Tensor<double> B("B", {NUM_I, NUM_J, NUM_K}, ecsr);
  Tensor<double> C("C", {NUM_I, NUM_J, NUM_K}, ecsr);
  Tensor<double> eA("eA", {NUM_I, NUM_J, NUM_K}, Dense);
  Tensor<double> eB("eB", {NUM_I, NUM_J, NUM_K}, Dense);
  Tensor<double> eC("eC", {NUM_I, NUM_J, NUM_K}, Dense);

  srand(75883);
  for (int i = 0; i < NUM_I; i++) {
    for (int j = 0; j < NUM_J; j++) {
      for (int k = 0; k < NUM_K; k++) {
        float rand_float = (float)rand()/(float)(RAND_MAX);
        if (rand_float < SPARSITY) {
          double val = (double)((int)(rand_float*3/SPARSITY));
          A.insert({i, j, k}, val);
          eA.insert({i, j, k}, val);
        }
      }
    }
  }

  for (int i = 0; i < NUM_I; i++) {
    for (int j = 0; j < NUM_J; j++) {
      for (int k = 0; k < NUM_K; k++) {
        float rand_float = (float)rand()/(float)(RAND_MAX);
        if (rand_float < SPARSITY) {
          double val = (double)((int)(rand_float*3/SPARSITY));
          B.insert({i, j, k}, val);
          eB.insert({i, j, k}, val);
        }
      }
    }
  }

  A.pack();
  B.pack();

  C(i, j, k) = A(i, j, k) + B(i, j, k);
  IndexStmt stmt = C.getAssignment().concretize();
  stmt = scheduleSpAddCPU(stmt);

  C.compile(stmt);
  C.assemble();
  C.compute();

  eC(i, j, k) = eA(i, j, k) + eB(i, j, k);
  eC.compile();
  eC.assemble();
  eC.compute();
  ASSERT_TENSOR_EQ(eC, C);
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

  //printToFile("sddmm_cpu", stmt);

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
  Tensor<double> x("x", {NUM_J}, Format({Dense}));
  Tensor<double> y("y", {NUM_I}, Format({Dense}));

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

  //printToFile("spmv_cpu", stmt);

  y.compile(stmt);
  y.assemble();
  y.compute();

  Tensor<double> expected("expected", {NUM_I}, Format({Dense}));
  expected(i) = A(i, j) * x(j);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, y);
}

TEST(scheduling_eval, precompute2D) {
  if (should_use_CUDA_codegen()) {
    return;
  }
  
  int NUM_I = 16;
  int NUM_J = 16;
  int NUM_K = 16;
  int NUM_L = 16;
  float SPARSITY = .3;
  Tensor<double> A("A", {NUM_I, NUM_J, NUM_K}, Format({Dense, Dense, Dense}));
  Tensor<double> x("x", {NUM_K, NUM_L}, Format({Dense, Dense}));
  Tensor<double> y("y", {NUM_I, NUM_J, NUM_L}, Format({Dense, Dense, Dense}));

  srand(120);
  for (int i = 0; i < NUM_I; i++) {
    for (int j = 0; j < NUM_J; j++) {
      for (int k = 0; k < NUM_K; k++) {
        float rand_float = (float)rand()/(float)(RAND_MAX);
        if (rand_float < SPARSITY) {
          A.insert({i, j, k}, (double) ((int) (rand_float * 3 / SPARSITY)));
        }
      }
    }
  }
  for (int k = 0; k < NUM_K; k++) {
    for (int l = 0; l < NUM_L; l++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      x.insert({k, l}, (double) ((int) (rand_float*3/SPARSITY)));
    }
  }

  x.pack();
  A.pack();


  IndexExpr precomputed = A(i, j, k) * x(k, l);
  y(i, j, l) = precomputed;

  IndexStmt stmt = y.getAssignment().concretize();
  stmt = schedulePrecompute3D(stmt, precomputed);

  y.compile(stmt);
  y.assemble();
  y.compute();

  Tensor<double> expected("expected", {NUM_I, NUM_J, NUM_L}, Format({Dense, Dense, Dense}));
  expected(i, j, l) = A(i, j, k) * x(k, l);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, y);
}

TEST(scheduling_eval, precompute1D) {
  if (should_use_CUDA_codegen()) {
    return;
  }
  int NUM_I = 1021/10;
  int NUM_J = 1039/10;
  float SPARSITY = .3;
  Tensor<double> A("A", {NUM_I, NUM_J}, CSR);
  Tensor<double> x("x", {NUM_J}, Format({Dense}));
  Tensor<double> y("y", {NUM_I}, Format({Dense}));

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

  IndexExpr precomputed = A(i, j) * x(j);
  y(i) = precomputed;

  IndexStmt stmt = y.getAssignment().concretize();
  stmt = schedulePrecompute1D(stmt, precomputed);

  y.compile(stmt);
  y.assemble();
  y.compute();

  Tensor<double> expected("expected", {NUM_I}, Format({Dense}));
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
  Tensor<double> c("c", {NUM_K}, Format({Dense}));

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

  //printToFile("ttv_cpu", stmt);

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

TEST(scheduling_eval, ttvCPU_CSR) {
  if (should_use_CUDA_codegen()) {
    return;
  }

  int NUM_I = 1021/10;
  int NUM_J = 1039/10;
  int NUM_K = 1057/10;
  float SPARSITY = .3;
  Tensor<double> A("A", {NUM_I, NUM_J}, {Dense, Sparse});
  Tensor<double> B("B", {NUM_I, NUM_J, NUM_K}, {Sparse, Sparse, Sparse});
  Tensor<double> c("c", {NUM_K}, Format({Dense}));

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
  stmt = scheduleTTVCPUCSR(stmt);

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

  //printToFile("ttm_cpu", stmt);

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
  Tensor<double> B("B", {NUM_I, NUM_K, NUM_L}, {Dense, Sparse, Sparse});
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
  //printToFile("mttkrp_cpu", stmt);

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
  Tensor<double> x("x", {NUM_J}, Format({Dense}));
  Tensor<double> y("y", {NUM_I}, Format({Dense}));

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

  //printToFile("spmv_gpu", stmt);

  y.compile(stmt);
  y.assemble();
  y.compute();

  Tensor<double> expected("expected", {NUM_I}, Format({Dense}));
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
  int NUM_K = 128;
  float SPARSITY = .3;
  Tensor<double> A("A", {NUM_I, NUM_J}, CSR);
  Tensor<double> B("B", {NUM_J, NUM_K}, {Dense, Dense});
  Tensor<double> C("C", {NUM_I, NUM_K}, Format({{Dense, Dense}, {1, 0}}));

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
  IndexExpr precomputed = A(i, j);
  C(i, k) = B(j, k) * precomputed;

  IndexStmt stmt = C.getAssignment().concretize();
  stmt = scheduleSpMMGPU(stmt, A, precomputed);

  //printToFile("spmm_gpu", stmt);

  C.compile(stmt);
  C.assemble();
  C.compute();

  Tensor<double> expected("expected", {NUM_I, NUM_K}, Format({{Dense, Dense}, {1, 0}}));
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

  //printToFile("spmm_dcsr_gpu", stmt);

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

  //printToFile("sddmm_gpu", stmt);

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

  //printToFile("ttm_gpu", stmt);

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
  Tensor<double> c("c", {NUM_K}, Format({Dense}));

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

  //printToFile("ttv_gpu", stmt);

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
  int NUM_J = 32;
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

  //printToFile("mttkrp_gpu", stmt);

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

TEST(scheduling_eval, indexVarSplit) {  
  
  Tensor<int> a("A", {4, 4}, dense);
  Tensor<int> b("B", {4, 4}, compressed);

  Tensor<int> expected("C", a.getDimensions(), Dense);
  const int n = a.getDimensions()[0];
  const int m = a.getDimensions()[1];

  for(int i = 0; i < n; ++i) {
    b.insert({i, i}, 2);
  }
  b.pack();

  a(i, j) = b(i, j) * (i * m + j);
  IndexStmt stmt = a.getAssignment().concretize();
  IndexVar j0("j0"), j1("j1");
  stmt = stmt.split(j, j0, j1, 2);

  a.compile(stmt);
  a.assemble();
  a.compute();

  for(int i = 0; i < n; ++i) {
    int flattened_idx = i * m + i;
    expected.insert({i, i}, 2 * flattened_idx);
  }
  expected.pack();

  ASSERT_TENSOR_EQ(expected, a);
}

TEST(generate_evaluation_files, DISABLED_cpu) {
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

  vector<vector<int>> mttkrp_parameters = {};
  for (int i = 1; i <= 64; i *= 2) {
    mttkrp_parameters.push_back({i,0});

  }
  vector<vector<int>> sddmm_parameters = {{16, 8}, {8, 8}};
  vector<vector<int>> ttv_parameters = {{16}, {8}, {32}};
  vector<vector<int>> ttm_parameters = {{16, 8}, {8, 8}};

  int NUM_I = 100;
  int NUM_J = 100;
  int NUM_K = 100;
  int NUM_L = 100;
  int NUM_M = 100;
  int NUM_N = 100;

  string file_ending = should_use_CUDA_codegen() ? ".cu" : ".c";
  string file_path = "eval_prepared_cpu/";
  mkdir(file_path.c_str(), 0777);

  // spmv
  {
    stringstream source;
    std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(source, ir::CodeGen::ImplementationGen);
    Tensor<double> A("A", {NUM_I, NUM_J}, CSR);
    Tensor<double> x("x", {NUM_J}, Format({Dense}));
    Tensor<double> y("y", {NUM_I}, Format({Dense}));
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
    Tensor<double> c("c", {NUM_K}, Format({Dense}));
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

  // mttkrp3
  {
    stringstream source;
    std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(source, ir::CodeGen::ImplementationGen);
    Tensor<double> A("A", {NUM_I, NUM_J}, {Dense, Dense});
    Tensor<double> B("B", {NUM_I, NUM_K, NUM_L}, {Dense, Sparse, Sparse});
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
    source_file.open(file_path + "mttkrp3_cpu" + file_ending);
    source_file << source.str();
    source_file.close();
  }

  // mttkrp3 workspace
  {
    stringstream source;
    std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(source, ir::CodeGen::ImplementationGen);
    Tensor<double> A("A", {NUM_I, NUM_J}, {Dense, Dense});
    Tensor<double> B("B", {NUM_I, NUM_K, NUM_L}, {Dense, Sparse, Sparse});
    Tensor<double> C("C", {NUM_K, NUM_J}, {Dense, Dense});
    Tensor<double> D("D", {NUM_L, NUM_J}, {Dense, Dense});
    IndexExpr precomputedExpr = B(i,k,l) * D(l,j);
    A(i,j) = precomputedExpr * C(k,j);
    IndexStmt stmt = A.getAssignment().concretize();
    TensorVar precomputed("precomputed", Type(Float64, {Dimension(j)}), taco::dense);

    IndexStmt precomputed_stmt = forall(i, forall(k,
                      where(forall(j, A(i,j) += precomputed(j) * C(k,j)),
                            forall(l, forall(j, precomputed(j) += B(i,k,l) * D(l,j))))));
    IndexStmt scheduled = scheduleMTTKRPPrecomputedCPU(precomputed_stmt, B, 64);
    ir::Stmt compute = lower(scheduled, string("mttkrp3_workspace"),  false, true);
    codegen->compile(compute, true);

    ofstream source_file;
    source_file.open(file_path + "mttkrp3_cpu_workspace" + file_ending);
    source_file << source.str();
    source_file.close();
  }

  // mttkrp4
  {
    stringstream source;
    std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(source, ir::CodeGen::ImplementationGen);
    Tensor<double> A("A", {NUM_I, NUM_J}, {Dense, Dense});
    Tensor<double> B("B", {NUM_I, NUM_K, NUM_L, NUM_M}, {Dense, Sparse, Sparse, Sparse});
    Tensor<double> C("C", {NUM_K, NUM_J}, {Dense, Dense});
    Tensor<double> D("D", {NUM_L, NUM_J}, {Dense, Dense});
    Tensor<double> E("E", {NUM_M, NUM_J}, {Dense, Dense});
    A(i,j) = B(i,k,l,m) * C(k,j) * D(l,j) * E(m,j);
    IndexStmt stmt = A.getAssignment().concretize();
    bool isFirst = true;
    for (auto paramSet : mttkrp_parameters) {
      IndexStmt scheduled = scheduleMTTKRP4CPU(stmt, B, paramSet[0], paramSet[1]);
      ir::Stmt compute = lower(scheduled, string("compute_") + util::join(paramSet, "_"),  false, true);
      codegen->compile(compute, isFirst);
      isFirst = false;
    }
    ofstream source_file;
    source_file.open(file_path + "mttkrp4_cpu" + file_ending);
    source_file << source.str();
    source_file.close();
  }

  // mttkrp4 workspace
  {
    stringstream source;
    std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(source, ir::CodeGen::ImplementationGen);
    Tensor<double> A("A", {NUM_I, NUM_J}, {Dense, Dense});
    Tensor<double> B("B", {NUM_I, NUM_K, NUM_L, NUM_M}, {Dense, Sparse, Sparse, Sparse});
    Tensor<double> C("C", {NUM_K, NUM_J}, {Dense, Dense});
    Tensor<double> D("D", {NUM_L, NUM_J}, {Dense, Dense});
    Tensor<double> E("E", {NUM_M, NUM_J}, {Dense, Dense});
    A(i,j) = B(i,k,l,m) * C(k,j) * D(l,j) * E(m,j);

    IndexExpr BE = B(i,k,l,m) * E(m,j);
    IndexExpr BDE = BE * D(l, j);
    A(i,j) = BDE * C(k,j);
    IndexStmt stmt = A.getAssignment().concretize();
    TensorVar BE_workspace("BE_workspace", Type(Float64, {Dimension(j)}), taco::dense);
    TensorVar BDE_workspace("BDE_workspace", Type(Float64, {Dimension(j)}), taco::dense);

    IndexStmt precomputed_stmt = forall(i, forall(k,
            where(forall(j, A(i,j) += BDE_workspace(j) * C(k,j)),
              forall(l, where(forall(j, BDE_workspace(j) += BE_workspace(j) * D(l,j)),
                  forall(m, forall(j, BE_workspace(j) += B(i,k,l,m) * E(m,j))))))));

    IndexStmt scheduled = scheduleMTTKRPPrecomputedCPU(precomputed_stmt, B, 64);
    ir::Stmt compute = lower(scheduled, string("mttkrp4_workspace"),  false, true);
    codegen->compile(compute, true);

    ofstream source_file;
    source_file.open(file_path + "mttkrp4_cpu_workspace" + file_ending);
    source_file << source.str();
    source_file.close();
  }

  // mttkrp5
  {
    stringstream source;
    std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(source, ir::CodeGen::ImplementationGen);
    Tensor<double> A("A", {NUM_I, NUM_J}, {Dense, Dense});
    Tensor<double> B("B", {NUM_I, NUM_K, NUM_L, NUM_M, NUM_N}, {Dense, Sparse, Sparse, Sparse, Sparse});
    Tensor<double> C("C", {NUM_K, NUM_J}, {Dense, Dense});
    Tensor<double> D("D", {NUM_L, NUM_J}, {Dense, Dense});
    Tensor<double> E("E", {NUM_M, NUM_J}, {Dense, Dense});
    Tensor<double> F("F", {NUM_N, NUM_J}, {Dense, Dense});
    A(i,j) = B(i,k,l,m,n) * C(k,j) * D(l,j) * E(m,j) * F(n,j);
    IndexStmt stmt = A.getAssignment().concretize();
    bool isFirst = true;
    for (auto paramSet : mttkrp_parameters) {
      IndexStmt scheduled = scheduleMTTKRP5CPU(stmt, B, paramSet[0], paramSet[1]);
      ir::Stmt compute = lower(scheduled, string("compute_") + util::join(paramSet, "_"),  false, true);
      codegen->compile(compute, isFirst);
      isFirst = false;
    }
    ofstream source_file;
    source_file.open(file_path + "mttkrp5_cpu" + file_ending);
    source_file << source.str();
    source_file.close();
  }

  // mttkrp5 workspace
  {
    stringstream source;
    std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(source, ir::CodeGen::ImplementationGen);
    Tensor<double> A("A", {NUM_I, NUM_J}, {Dense, Dense});
    Tensor<double> B("B", {NUM_I, NUM_K, NUM_L, NUM_M, NUM_N}, {Dense, Sparse, Sparse, Sparse, Sparse});
    Tensor<double> C("C", {NUM_K, NUM_J}, {Dense, Dense});
    Tensor<double> D("D", {NUM_L, NUM_J}, {Dense, Dense});
    Tensor<double> E("E", {NUM_M, NUM_J}, {Dense, Dense});
    Tensor<double> F("F", {NUM_N, NUM_J}, {Dense, Dense});
    A(i,j) = B(i,k,l,m,n) * C(k,j) * D(l,j) * E(m,j) * F(n,j);
    IndexStmt stmt = A.getAssignment().concretize();

    IndexExpr BF = B(i,k,l,m,n) * F(n,j);
    IndexExpr BEF = BF * E(m,j);
    IndexExpr BDEF = BEF * D(l, j);
    A(i,j) = BDEF * C(k,j);
    TensorVar BF_workspace("BF_workspace", Type(Float64, {Dimension(j)}), taco::dense);
    TensorVar BEF_workspace("BEF_workspace", Type(Float64, {Dimension(j)}), taco::dense);
    TensorVar BDEF_workspace("BDEF_workspace", Type(Float64, {Dimension(j)}), taco::dense);

    IndexStmt precomputed_stmt = forall(i, forall(k,
            where(forall(j, A(i,j) += BDEF_workspace(j) * C(k,j)),
               forall(l, where(forall(j, BDEF_workspace(j) += BEF_workspace(j) * D(l,j)),
                 forall(m, where(forall(j, BEF_workspace(j) += BF_workspace(j) * E(m,j)),
                   forall(n, forall(j, BF_workspace(j) += B(i,k,l,m,n)*F(n,j))))))))));

    IndexStmt scheduled = scheduleMTTKRPPrecomputedCPU(precomputed_stmt, B, 64);
    ir::Stmt compute = lower(scheduled, string("mttkrp5_workspace"),  false, true);
    codegen->compile(compute, true);

    ofstream source_file;
    source_file.open(file_path + "mttkrp5_cpu_workspace" + file_ending);
    source_file << source.str();
    source_file.close();
  }
}

TEST(generate_evaluation_files, DISABLED_gpu) {
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
    spmm_parameters.push_back({i,512});
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
    Tensor<double> x("x", {NUM_J}, Format({Dense}));
    Tensor<double> y("y", {NUM_I}, Format({Dense}));
    IndexExpr precomputed = A(i, j) * x(j);
    y(i) = precomputed;
    IndexStmt stmt = y.getAssignment().concretize();
    bool isFirst = true;

    IndexStmt scheduled = scheduleSpMVRowsGPU(stmt, A, precomputed);
    ir::Stmt compute = lower(scheduled, string("compute_warp_row"),  false, true);
    codegen->compile(compute, isFirst);
    isFirst = false;

    scheduled = scheduleSpMVThreadPerRowGPU(stmt, A, precomputed);
    compute = lower(scheduled, string("compute_thread_row"),  false, true);
    codegen->compile(compute, isFirst);


    ofstream source_file;
    source_file.open(file_path + "spmv_gpu_warp_vs_thread" + file_ending);
    source_file << source.str();
    source_file.close();
  }

  // spmv
  {
    stringstream source;
    std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(source, ir::CodeGen::ImplementationGen);
    Tensor<double> A("A", {NUM_I, NUM_J}, CSR);
    Tensor<double> x("x", {NUM_J}, Format({Dense}));
    Tensor<double> y("y", {NUM_I}, Format({Dense}));
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
      int NUM_K = 128;
      Tensor<double> B("B", {NUM_J, NUM_K}, {Dense, Dense});
      Tensor<double> C("C", {NUM_I, NUM_K}, Format({{Dense, Dense}, {1, 0}}));
      IndexExpr precomputed = A(i, j);
      C(i, k) = precomputed * B(j, k);
      IndexStmt stmt = C.getAssignment().concretize();
      IndexStmt scheduled = scheduleSpMMGPU(stmt, A, precomputed, paramSet[0], paramSet[1]);
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
    Tensor<double> c("c", {NUM_K}, Format({Dense}));
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
      int NUM_J = WARP_SIZE;
      Tensor<double> A("A", {NUM_I, NUM_J}, {Dense, Dense});
      Tensor<double> C("C", {NUM_K, NUM_J}, {Dense, Dense});
      Tensor<double> D("D", {NUM_L, NUM_J}, {Dense, Dense});
      A(i,j) = B(i,k,l) * C(k,j) * D(l,j);
      IndexStmt stmt = A.getAssignment().concretize();
      IndexStmt scheduled = scheduleMTTKRPGPU(stmt, B, paramSet[0], paramSet[1]);
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

TEST(generate_figures, DISABLED_cpu) {
  if (should_use_CUDA_codegen()) {
    return;
  }

  int NUM_I = 100;
  int NUM_J = 100;

  string file_ending = should_use_CUDA_codegen() ? ".cu" : ".c";
  string file_path = "figures_cpu/";
  mkdir(file_path.c_str(), 0777);

  // spmv
  {
    stringstream source;
    std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(source, ir::CodeGen::ImplementationGen);
    Tensor<double> A("A", {NUM_I, NUM_J}, CSR);
    Tensor<double> x("x", {NUM_J}, Format({Dense}));
    Tensor<double> y("y", {NUM_I}, Format({Dense}));
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

TEST(scheduling_eval, DISABLED_bfsPullScheduled) {
  if (should_use_CUDA_codegen()) {
    return;
  }
  constexpr int numVertices = 10;
  int NUM_I = numVertices;
  int NUM_J = numVertices;
  float SPARSITY = .3;

  Tensor<uint16_t> A("A", {NUM_I, NUM_J}, CSC);
  Tensor<uint16_t> x("x", {NUM_J}, {Sparse});
  Tensor<uint16_t> m("mask", {NUM_J}, {Dense});
  Tensor<uint16_t> y("y", {NUM_I}, {Dense});
  Tensor<int> step("step");

  uint16_t one = 1;
  uint16_t zero = 0;

  Func scOr("Or", OrImpl(), {Annihilator(one), Identity(zero)});
  Func scAnd("And", AndImpl(), {Annihilator(zero), Identity(one)});
  Func bfsMaskOp("bfsMask", BfsLower(), BfsMaskAlg());

  srand(120);
  for (int i = 0; i < NUM_I; i++) {
    for (int j = 0; j < NUM_J; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      if (rand_float < SPARSITY) {
        A.insert({i, j}, one);
      }
    }
  }

  for (int j = 0; j < NUM_J; j++) {
    float rand_float = (float)rand()/(float)(RAND_MAX);
    if (rand_float < SPARSITY) {
      x.insert({j}, one);
    } else {
      x.insert({j}, zero);
    }
  }

  x.pack();
  A.pack();
  m.pack();

  y(i) = Reduction(scOr(), j, scAnd(A(i, j), x(j)));
  IndexStmt stmt = y.getAssignment().concretize();

  stmt = stmt.reorder(i,j)
             .parallelize(j, taco::ParallelUnit::CPUThread, taco::OutputRaceStrategy::Atomics);
  printToFile("bfs_push", stmt);
  y.compile(stmt);
  y.assemble();
  y.compute();

  Tensor<uint16_t> s("s", {NUM_J}, {Sparse});
  Tensor<uint16_t> d("d", {NUM_J}, {dense});

  Func sparsifyOp("sparsify", identityFunc(), ComplementUnion());
  s(i) = sparsifyOp(d(i), i);
  IndexStmt sparsify = s.getAssignment().concretize();
  printToFile("sparsify", sparsify);


  Tensor<uint16_t> expected("expected", {NUM_I}, {Dense});
  expected(i) = Reduction(scOr(), j, bfsMaskOp(scAnd(A(i, j), x(j)), m(i)));
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, y);
}
