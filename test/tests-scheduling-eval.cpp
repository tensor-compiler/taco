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
const IndexVar i("i"), j("j"), k("k");

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


  IndexVar i("i"), j("j"), k("k");
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

  IndexVar i("i"), j("j"), k("k");
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
  int NUM_K = 1057/10;
  float SPARSITY = .3;
  int UNROLL_FACTOR = 8;
  int CHUNK_SIZE = 16;
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

  IndexVar i("i"), j("j"), k("k");
  IndexVar i0("i0"), i1("i1"), jpos("jpos"), jpos0("jpos0"), jpos1("jpos1");
  C(i, k) = A(i, j) * B(j, k);

  IndexStmt stmt = C.getAssignment().concretize();
  stmt = stmt.split(i, i0, i1, CHUNK_SIZE)
          .pos(j, jpos, A(i,j))
          .split(jpos, jpos0, jpos1, UNROLL_FACTOR)
          .reorder({i0, i1, jpos0, k, jpos1})
          .parallelize(i0, PARALLEL_UNIT::CPU_THREAD, OUTPUT_RACE_STRATEGY::NO_RACES)
          .parallelize(jpos1, PARALLEL_UNIT::CPU_VECTOR, OUTPUT_RACE_STRATEGY::IGNORE_RACES);

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
  int UNROLL_FACTOR = 8;
  int CHUNK_SIZE = 16;
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

  IndexVar i("i"), j("j"), k("k");
  IndexVar i0("i0"), i1("i1"), kpos("kpos"), kpos0("kpos0"), kpos1("kpos1");
  A(i,k) = B(i,k) * C(i,j) * D(j,k);

  IndexStmt stmt = A.getAssignment().concretize();
  stmt = stmt.split(i, i0, i1, CHUNK_SIZE)
          .pos(k, kpos, B(i,k))
          .split(kpos, kpos0, kpos1, UNROLL_FACTOR)
          .reorder({i0, i1, kpos0, j, kpos1})
          .parallelize(i0, PARALLEL_UNIT::CPU_THREAD, OUTPUT_RACE_STRATEGY::NO_RACES)
          .parallelize(kpos1, PARALLEL_UNIT::CPU_VECTOR, OUTPUT_RACE_STRATEGY::IGNORE_RACES);

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
  int CHUNK_SIZE = 16;
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

  IndexVar i("i"), j("j"), k("k");
  IndexVar i0("i0"), i1("i1"), kpos("kpos"), kpos0("kpos0"), kpos1("kpos1");
  y(i) = A(i, j) * x(j);

  IndexStmt stmt = y.getAssignment().concretize();
  stmt = stmt.split(i, i0, i1, CHUNK_SIZE)
          .reorder({i0, i1, j})
          .parallelize(i0, PARALLEL_UNIT::CPU_THREAD, OUTPUT_RACE_STRATEGY::NO_RACES);

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
  int CHUNK_SIZE = 16;
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

  IndexVar i("i"), j("j"), k("k");
  IndexVar f("f"), fpos("fpos"), chunk("chunk"), fpos2("fpos2");
  A(i,j) = B(i,j,k) * c(k);

  IndexStmt stmt = A.getAssignment().concretize();
  stmt = stmt.fuse(i, j, f)
          .pos(f, fpos, B(i,j,k))
          .split(fpos, chunk, fpos2, CHUNK_SIZE)
          .reorder({chunk, fpos2, k})
          .parallelize(chunk, PARALLEL_UNIT::CPU_THREAD, OUTPUT_RACE_STRATEGY::NO_RACES);

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
  int CHUNK_SIZE = 16;
  int UNROLL_FACTOR = 8;
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

  IndexVar i("i"), j("j"), k("k"), l("l");
  IndexVar f("f"), fpos("fpos"), chunk("chunk"), fpos2("fpos2"), kpos("kpos"), kpos1("kpos1"), kpos2("kpos2");
  A(i,j,l) = B(i,j,k) * C(k,l);

  IndexStmt stmt = A.getAssignment().concretize();
  stmt = stmt.fuse(i, j, f)
          .pos(f, fpos, B(i,j,k))
          .split(fpos, chunk, fpos2, CHUNK_SIZE)
          .pos(k, kpos, B(i,j,k))
          .split(kpos, kpos1, kpos2, UNROLL_FACTOR)
          .reorder({chunk, fpos2, kpos1, l, kpos2})
          .parallelize(chunk, PARALLEL_UNIT::CPU_THREAD, OUTPUT_RACE_STRATEGY::NO_RACES)
          .parallelize(kpos2, PARALLEL_UNIT::CPU_VECTOR, OUTPUT_RACE_STRATEGY::IGNORE_RACES);

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
  int CHUNK_SIZE = 16;
  int UNROLL_FACTOR = 8;
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

  IndexVar i("i"), j("j"), k("k"), l("l");
  IndexVar f("f"), fpos("fpos"), chunk("chunk"), fpos2("fpos2"), lpos("lpos"), lpos1("lpos1"), lpos2("lpos2");
  A(i,j) = B(i,k,l) * C(k,j) * D(l,j);

  IndexStmt stmt = A.getAssignment().concretize();
  stmt = stmt.reorder({i,k,j,l}) // TODO: this shouldn't be necessary
          .fuse(i, k, f)
          .pos(f, fpos, B(i,k,l))
          .split(fpos, chunk, fpos2, CHUNK_SIZE)
          .pos(l, lpos, B(i,k,l))
          .split(lpos, lpos1, lpos2, UNROLL_FACTOR)
          .reorder({chunk, fpos2, lpos1, j, lpos2})
          .parallelize(chunk, PARALLEL_UNIT::CPU_THREAD, OUTPUT_RACE_STRATEGY::ATOMICS)
          .parallelize(lpos2, PARALLEL_UNIT::CPU_VECTOR, OUTPUT_RACE_STRATEGY::IGNORE_RACES);

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
  int NNZ_PER_THREAD = 8;
  int BLOCK_SIZE = 256;
  int WARP_SIZE = 32;
  int NNZ_PER_WARP = NNZ_PER_THREAD * WARP_SIZE;
  int NNZ_PER_TB = NNZ_PER_THREAD * BLOCK_SIZE;
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

  IndexVar i("i"), j("j");
  IndexVar f("f"), fpos("fpos"), fpos1("fpos1"), fpos2("fpos2"), block("block"), warp("warp"), thread("thread"), thread_nz("thread_nz");
  y(i) = A(i, j) * x(j);

  IndexStmt stmt = y.getAssignment().concretize();
  stmt = stmt.fuse(i, j, f)
          .pos(f, fpos, A(i, j))
          .split(fpos, block, fpos1, NNZ_PER_TB)
          .split(fpos1, warp, fpos2, NNZ_PER_WARP)
          .split(fpos2, thread, thread_nz, NNZ_PER_THREAD)
          .reorder({block, warp, thread, thread_nz})
          .parallelize(block, PARALLEL_UNIT::GPU_BLOCK, OUTPUT_RACE_STRATEGY::IGNORE_RACES)
          .parallelize(warp, PARALLEL_UNIT::GPU_WARP, OUTPUT_RACE_STRATEGY::IGNORE_RACES)
          .parallelize(thread, PARALLEL_UNIT::GPU_THREAD, OUTPUT_RACE_STRATEGY::ATOMICS); // TODO: TEMPORARY -> PARALLEL_REDUCTION

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
  int NUM_K = 128;
  float SPARSITY = .3;
  int WARP_SIZE = 32;
  int NNZ_PER_WARP = 8 * WARP_SIZE;
  int BLOCK_SIZE = 256;
  int NNZ_PER_TB = NNZ_PER_WARP * (BLOCK_SIZE / WARP_SIZE);
  int CO_FACTOR = 4;
  taco_iassert(NUM_K == CO_FACTOR * WARP_SIZE);
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

  IndexVar i("i"), j("j"), k("k");
  IndexVar f("f"), fpos("fpos"), block("block"), fpos1("fpos1"), warp("warp"), nnz("nnz");
  IndexVar dense_val("dense_val"), thread("thread");
  IndexVar thread_nz("thread_nz");
  C(i, k) = A(i, j) * B(j, k);

  IndexStmt stmt = C.getAssignment().concretize();
  stmt = stmt.reorder({i, j, k})
          .fuse(i, j, f)
          .pos(f, fpos, A(i, j))
          .split(fpos, block, fpos1, NNZ_PER_TB)
          .split(fpos1, warp, nnz, NNZ_PER_WARP)
          .split(k, dense_val, thread, WARP_SIZE)
          .reorder({block, warp, nnz, thread, dense_val})
          .parallelize(block, PARALLEL_UNIT::GPU_BLOCK, OUTPUT_RACE_STRATEGY::ATOMICS)
          .parallelize(warp, PARALLEL_UNIT::GPU_WARP, OUTPUT_RACE_STRATEGY::IGNORE_RACES)
          .parallelize(thread, PARALLEL_UNIT::GPU_THREAD, OUTPUT_RACE_STRATEGY::IGNORE_RACES); // TODO: TEMPORARY -> PARALLEL_REDUCTION

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

TEST(scheduling_eval, sddmmGPU) {
  if (!should_use_CUDA_codegen()) {
    return;
  }
  int NUM_I = 1021/10;
  int NUM_J = 1039/10;
  int NUM_K = 128;
  float SPARSITY = .3;
  int WARP_SIZE = 32;
  int NNZ_PER_WARP = 8 * WARP_SIZE;
  int BLOCK_SIZE = 256;
  int NNZ_PER_TB = NNZ_PER_WARP * (BLOCK_SIZE / WARP_SIZE);
  int CO_FACTOR = 4;
  taco_iassert(NUM_K == CO_FACTOR * WARP_SIZE);
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

  IndexVar i("i"), j("j"), k("k");
  IndexVar f("f"), fpos("fpos"), block("block"), fpos1("fpos1"), warp("warp"), nnz("nnz");
  IndexVar dense_val("dense_val"), thread("thread");
  IndexVar thread_nz("thread_nz");
  A(i,k) = B(i,k) * C(i,j) * D(j,k);

  IndexStmt stmt = A.getAssignment().concretize();
  stmt = stmt.reorder({i, k, j})
          .fuse(i, k, f)
          .pos(f, fpos, B(i,k))
          .split(fpos, block, fpos1, NNZ_PER_TB)
          .split(fpos1, warp, nnz, NNZ_PER_WARP)
          .split(j, dense_val, thread, WARP_SIZE)
          .reorder({block, warp, nnz, thread, dense_val})
          .parallelize(block, PARALLEL_UNIT::GPU_BLOCK, OUTPUT_RACE_STRATEGY::ATOMICS)
          .parallelize(warp, PARALLEL_UNIT::GPU_WARP, OUTPUT_RACE_STRATEGY::IGNORE_RACES)
          .parallelize(thread, PARALLEL_UNIT::GPU_THREAD, OUTPUT_RACE_STRATEGY::IGNORE_RACES); // TODO: TEMPORARY -> PARALLEL_REDUCTION

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
  int NUM_K = 1057/40;
  int NUM_L = 1232/40;
  float SPARSITY = .1;
  int CHUNK_SIZE = 16;
  int UNROLL_FACTOR = 8;

  int WARP_SIZE = 32;
  int NNZ_PER_WARP = 8 * WARP_SIZE;
  int BLOCK_SIZE = 256;
  int NNZ_PER_TB = NNZ_PER_WARP * (BLOCK_SIZE / WARP_SIZE);
  int CO_FACTOR = 4;
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

  IndexVar i("i"), j("j"), k("k"), l("l");
  IndexVar jk("jk"), f("f"), fpos("fpos"), block("block"), fpos1("fpos1"), warp("warp"), nnz("nnz"), dense_val("dense_val"), thread("thread");
  A(i,j,l) = B(i,j,k) * C(k,l);

  IndexStmt stmt = A.getAssignment().concretize();
  stmt = stmt.reorder({i, j, k, l})
          .fuse(j, k, jk)
          .fuse(i, jk, f)
          .pos(f, fpos, B(i, j, k))
          .split(fpos, block, fpos1, NNZ_PER_TB)
          .split(fpos1, warp, nnz, NNZ_PER_WARP)
          .split(l, dense_val, thread, WARP_SIZE)
          .reorder({block, warp, nnz, thread, dense_val})
          .parallelize(block, PARALLEL_UNIT::GPU_BLOCK, OUTPUT_RACE_STRATEGY::ATOMICS)
          .parallelize(warp, PARALLEL_UNIT::GPU_WARP, OUTPUT_RACE_STRATEGY::IGNORE_RACES)
          .parallelize(thread, PARALLEL_UNIT::GPU_THREAD, OUTPUT_RACE_STRATEGY::IGNORE_RACES);

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
  int CHUNK_SIZE = 16;
  int WARP_SIZE = 32;
  int NNZ_PER_WARP = 8 * WARP_SIZE;
  int BLOCK_SIZE = 256;
  int NNZ_PER_TB = NNZ_PER_WARP * (BLOCK_SIZE / WARP_SIZE);
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

  IndexVar i("i"), j("j"), k("k");
  IndexVar jk("jk"), f("f"), fpos("fpos"), block("block"), fpos1("fpos1"), warp("warp"), fpos2("fpos2"), thread("thread"), thread_nz("thread_nz");
  A(i,j) = B(i,j,k) * c(k);

  IndexStmt stmt = A.getAssignment().concretize();
  stmt = stmt.fuse(j, k, jk)
          .fuse(i, jk, f)
          .pos(f, fpos, B(i,j,k))
          .split(fpos, block, fpos1, NNZ_PER_TB)
          .split(fpos1, warp, fpos2, NNZ_PER_WARP)
          .split(fpos2, thread, thread_nz, NNZ_PER_WARP/WARP_SIZE)
          .reorder({block, warp, thread, thread_nz})
          .parallelize(block, PARALLEL_UNIT::GPU_BLOCK, OUTPUT_RACE_STRATEGY::IGNORE_RACES)
          .parallelize(warp, PARALLEL_UNIT::GPU_WARP, OUTPUT_RACE_STRATEGY::IGNORE_RACES)
          .parallelize(thread, PARALLEL_UNIT::GPU_THREAD, OUTPUT_RACE_STRATEGY::ATOMICS); // TODO: TEMPORARY -> PARALLEL_REDUCTION

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