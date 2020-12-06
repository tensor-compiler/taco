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

TEST(scheduling, splitEquality) {
  IndexVar i1, i2;
  IndexVar j1, j2;
  IndexVarRel rel1 = IndexVarRel(new SplitRelNode(i, i1, i2, 2));
  IndexVarRel rel2 = IndexVarRel(new SplitRelNode(i, i1, i2, 2));
  IndexVarRel rel3 = IndexVarRel(new SplitRelNode(j, i1, i1, 2));
  IndexVarRel rel4 = IndexVarRel(new SplitRelNode(i, i1, i2, 4));
  IndexVarRel rel5 = IndexVarRel(new SplitRelNode(i, j1, j2, 2));

  ASSERT_EQ(rel1, rel2);
  ASSERT_NE(rel1, rel3);
  ASSERT_NE(rel1, rel4);
  ASSERT_NE(rel1, rel5);
}

TEST(scheduling, forallReplace) {
  IndexVar i1, j1, j2;
  Type t(type<double>(), {3});
  TensorVar a("a", t), b("b", t);
  IndexStmt stmt = forall(i, forall(i1, a(i) = b(i)));
  IndexStmt replaced = Transformation(ForAllReplace({i, i1}, {j, j1, j2})).apply(stmt);
  ASSERT_NE(stmt, replaced);

  ASSERT_TRUE(isa<Forall>(replaced));
  Forall jForall = to<Forall>(replaced);
  ASSERT_EQ(j, jForall.getIndexVar());

  ASSERT_TRUE(isa<Forall>(jForall.getStmt()));
  Forall j1Forall = to<Forall>(jForall.getStmt());
  ASSERT_EQ(j1, j1Forall.getIndexVar());

  ASSERT_TRUE(isa<Forall>(j1Forall.getStmt()));
  Forall j2Forall = to<Forall>(j1Forall.getStmt());
  ASSERT_EQ(j2, j2Forall.getIndexVar());

  ASSERT_TRUE(equals(a(i) = b(i), j2Forall.getStmt()));
  ASSERT_TRUE(equals(forall(j, forall(j1, forall(j2, a(i) = b(i)))), replaced));
}

TEST(scheduling, splitIndexStmt) {
  Type t(type<double>(), {3});
  TensorVar a("a", t), b("b", t);
  IndexVar i1, i2;
  IndexStmt stmt = forall(i, a(i) = b(i));
  IndexStmt splitStmt = stmt.split(i, i1, i2, 2);

  ASSERT_TRUE(isa<SuchThat>(splitStmt));
  SuchThat suchThat = to<SuchThat>(splitStmt);
  ASSERT_EQ(suchThat.getPredicate(), vector<IndexVarRel>({IndexVarRel(new SplitRelNode(i, i1, i2, 2))}));

  ASSERT_TRUE(isa<Forall>(suchThat.getStmt()));
  Forall i1Forall = to<Forall>(suchThat.getStmt());
  ASSERT_EQ(i1, i1Forall.getIndexVar());

  ASSERT_TRUE(isa<Forall>(i1Forall.getStmt()));
  Forall i2Forall = to<Forall>(i1Forall.getStmt());
  ASSERT_EQ(i2, i2Forall.getIndexVar());

  ASSERT_TRUE(equals(a(i) = b(i), i2Forall.getStmt()));
}

TEST(scheduling, lowerDenseMatrixMul) {
  Tensor<double> A("A", {4, 4}, {Dense, Dense});
  Tensor<double> B("B", {4, 4}, {Dense, Dense});
  Tensor<double> C("C", {4, 4}, {Dense, Dense});

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      A.insert({i, j}, (double) i+j);
      B.insert({i, j}, (double) i+j);
    }
  }

  A.pack();
  B.pack();

  IndexVar i("i"), j("j"), k("k");
  IndexVar i0("i0"), i1("i1"), j0("j0"), j1("j1"), k0("k0"), k1("k1");
  C(i, j) = A(i, k) * B(k, j);

  IndexStmt stmt = C.getAssignment().concretize();
  stmt = stmt.split(i, i0, i1, 2)
             .split(j, j0, j1, 2)
             .split(k, k0, k1, 2)
             .reorder({i0, j0, k0, i1, j1, k1});

  C.compile(stmt);
  C.assemble();
  C.compute();

  Tensor<double> expected("expected", {4, 4}, {Dense, Dense});
  expected(i, j) = A(i, k) * B(k, j);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(C, expected);

  //  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);
  //  ir::Stmt compute = lower(stmt, "compute",  false, true);
  //  codegen->compile(compute, true);
}

TEST(scheduling, lowerSparseCopy) {
  Tensor<double> A("A", {8}, {Sparse});
  Tensor<double> C("C", {8}, {Dense});

  for (int i = 0; i < 8; i++) {
    if (i % 2 == 0) {
      A.insert({i}, (double) i);
    }
  }

  A.pack();

  IndexVar i("i");
  IndexVar i0("i0"), i1("i1");
  C(i) = A(i);

  IndexStmt stmt = C.getAssignment().concretize();
  stmt = stmt.split(i, i0, i1, 4);

  C.compile(stmt);
  C.assemble();
  C.compute();

  Tensor<double> expected("expected", {8}, {Dense});
  expected(i) = A(i);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, C);

  //  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);
  //  ir::Stmt compute = lower(stmt, "compute",  false, true);
  //  codegen->compile(compute, true);
}

TEST(scheduling, lowerSparseMulDense) {
  Tensor<double> A("A", {8}, {Sparse});
  Tensor<double> B("B", {8}, {Dense});
  Tensor<double> C("C", {8}, {Dense});

  for (int i = 0; i < 8; i++) {
    if (i % 2 == 0) {
      A.insert({i}, (double) i);
    }
    B.insert({i}, (double) i);
  }

  A.pack();
  B.pack();

  IndexVar i("i");
  IndexVar i0("i0"), i1("i1");
  C(i) = A(i) * B(i);

  IndexStmt stmt = C.getAssignment().concretize();
  stmt = stmt.split(i, i0, i1, 4);

  C.compile(stmt);
  C.assemble();
  C.compute();

  Tensor<double> expected("expected", {8}, {Dense});
  expected(i) = A(i) * B(i);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, C);

  //  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);
  //  ir::Stmt compute = lower(stmt, "compute",  false, true);
  //  codegen->compile(compute, true);
}

TEST(scheduling, lowerSparseMulSparse) {
  Tensor<double> A("A", {8}, {Sparse});
  Tensor<double> B("B", {8}, {Sparse});
  Tensor<double> C("C", {8}, {Dense});

  for (int i = 0; i < 8; i++) {
    if (i % 2 == 0) {
      A.insert({i}, (double) i);
    }
    if (i != 2 && i != 3 && i != 4) {
      B.insert({i}, (double) i);
    }
  }

  A.pack();
  B.pack();

  IndexVar i("i");
  IndexVar i0("i0"), i1("i1");
  C(i) = A(i) * B(i);

  IndexStmt stmt = C.getAssignment().concretize();
  stmt = stmt.split(i, i0, i1, 4);

  C.compile(stmt);
  C.assemble();
  C.compute();

  Tensor<double> expected("expected", {8}, {Dense});
  expected(i) = A(i) * B(i);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, C);

  //  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);
  //  ir::Stmt compute = lower(stmt, "compute",  false, true);
  //  codegen->compile(compute, true);
}

TEST(scheduling, lowerSparseAddSparse) {
  Tensor<double> A("A", {8}, {Sparse});
  Tensor<double> B("B", {8}, {Sparse});
  Tensor<double> C("C", {8}, {Dense});

  for (int i = 0; i < 8; i++) {
    if (i % 2 == 0) {
      A.insert({i}, (double) i);
    }
    if (i != 2 && i != 3 && i != 4) {
      B.insert({i}, (double) i);
    }
  }

  A.pack();
  B.pack();

  IndexVar i("i");
  IndexVar i0("i0"), i1("i1");
  C(i) = A(i) + B(i);

  IndexStmt stmt = C.getAssignment().concretize();
  stmt = stmt.split(i, i0, i1, 4);

  C.compile(stmt);
  C.assemble();
  C.compute();

  Tensor<double> expected("expected", {8}, {Dense});
  expected(i) = A(i) + B(i);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, C);

  //  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);
  //  ir::Stmt compute = lower(stmt, "compute",  false, true);
  //  codegen->compile(compute, true);
}


TEST(scheduling, lowerSparseMatrixMul) {
  Tensor<double> A("A", {8, 8}, CSR);
  Tensor<double> B("B", {8, 8}, CSC);
  Tensor<double> C("C", {8, 8}, {Dense, Dense});

  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 8; j++) {
      if ((i+j) % 2 == 0) {
        A.insert({i, j}, (double) (i+j));
      }
      if ((i+j) != 2 && (i+j) != 3 && (i+j) != 4) {
        B.insert({i, j}, (double) (i+j));
      }
    }
  }

  A.pack();
  B.pack();

  IndexVar i("i"), j("j"), k("k");
  IndexVar i0("i0"), i1("i1"), j0("j0"), j1("j1"), k0("k0"), k1("k1");
  C(i, j) = A(i, k) * B(k, j);

  IndexStmt stmt = C.getAssignment().concretize();
  stmt = stmt.split(i, i0, i1, 2)
          .split(j, j0, j1, 2)
          .split(k, k0, k1, 2)
          .reorder({i0, j0, k0, i1, j1, k1})
          .parallelize(i0, should_use_CUDA_codegen() ? ParallelUnit::GPUBlock : ParallelUnit::CPUThread, should_use_CUDA_codegen() ? OutputRaceStrategy::IgnoreRaces : OutputRaceStrategy::Atomics);

  if (should_use_CUDA_codegen()) {
    stmt = stmt.parallelize(j0, ParallelUnit::GPUThread, OutputRaceStrategy::Atomics);
  }

  C.compile(stmt);
  C.assemble();
  C.compute();

  Tensor<double> expected("expected", {8, 8}, {Dense, Dense});
  expected(i, j) = A(i, k) * B(k, j);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, C);

//  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);
//  ir::Stmt compute = lower(stmt, "compute",  false, true);
//  codegen->compile(compute, true);
}

TEST(scheduling, parallelizeAtomicReduction) {
  Tensor<double> A("A", {8}, {Sparse});
  Tensor<double> B("B", {8}, {Dense});
  Tensor<double> C("C");

  for (int i = 0; i < 8; i++) {
    if (i % 2 == 0) {
      A.insert({i}, (double) i);
    }
    B.insert({i}, (double) i);
  }

  A.pack();
  B.pack();

  IndexVar i("i");
  IndexVar block("block"), thread("thread"), i0("i0"), i1("i1");
  C = A(i) * B(i);

  IndexStmt stmt = C.getAssignment().concretize();
  if (should_use_CUDA_codegen()) {
    stmt = stmt.split(i, i0, i1, 2)
            .split(i0, block, thread, 2)
            .parallelize(block, ParallelUnit::GPUBlock, OutputRaceStrategy::Atomics)
            .parallelize(thread, ParallelUnit::GPUThread, OutputRaceStrategy::Atomics);
  }
  else {
    stmt = stmt.split(i, i0, i1, 2)
            .parallelize(i0, ParallelUnit::CPUThread, OutputRaceStrategy::Atomics);
  }

  C.compile(stmt);
  C.assemble();
  C.compute();

  Tensor<double> expected("expected");
  expected = A(i) * B(i);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, C);

//  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);
//  ir::Stmt compute = lower(stmt, "compute",  false, true);
//  codegen->compile(compute, true);
}

TEST(scheduling, parallelizeTemporaryReduction) {
  Tensor<double> A("A", {8}, {Sparse});
  Tensor<double> B("B", {8}, {Dense});
  Tensor<double> C("C");

  for (int i = 0; i < 8; i++) {
    if (i % 2 == 0) {
      A.insert({i}, (double) i);
    }
    B.insert({i}, (double) i);
  }

  A.pack();
  B.pack();

  IndexVar i("i");
  IndexVar block("block"), thread("thread"), i0("i0"), i1("i1");
  C = A(i) * B(i);

  IndexStmt stmt = C.getAssignment().concretize();
  if (should_use_CUDA_codegen()) {
    stmt = stmt.split(i, i0, i1, 2)
            .split(i0, block, thread, 2)
            .parallelize(block, ParallelUnit::GPUBlock, OutputRaceStrategy::Temporary)
            .parallelize(thread, ParallelUnit::GPUThread, OutputRaceStrategy::Temporary);
  }
  else {
    stmt = stmt.split(i, i0, i1, 2)
            .parallelize(i0, ParallelUnit::CPUThread, OutputRaceStrategy::Temporary);
  }

  C.compile(stmt);
  C.assemble();
  C.compute();

  Tensor<double> expected("expected");
  expected = A(i) * B(i);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, C);

//  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);
//  ir::Stmt compute = lower(stmt, "compute",  false, true);
//  codegen->compile(compute, true);
}

TEST(scheduling, multilevel_tiling) {
  Tensor<double> A("A", {8}, {Sparse});
  Tensor<double> B("B", {8}, {Sparse});
  Tensor<double> C("C", {8}, {Dense});

  for (int i = 0; i < 8; i++) {
    A.insert({i}, (double) i);
    //if (i != 2 && i != 3 && i != 4) {
      B.insert({i}, (double) i);
    //}
  }

  A.pack();
  B.pack();

  Tensor<double> expected("expected", {8}, {Dense});
  expected(i) = A(i) * B(i);
  expected.compile();
  expected.assemble();
  expected.compute();

  IndexVar i("i");
  IndexVar iX("iX"), iX1("iX1"), iX2("iX2"), iY("iY"), iY1("iY1"), iY2("iY2");
  C(i) = A(i) * B(i);

  IndexStmt stmt = C.getAssignment().concretize();
  stmt = stmt.split(i, iX, iY, 2)
          .split(iX, iX1, iX2, 2)
          .split(iY, iY1, iY2, 2);

  vector<IndexVar> reordering = {iX1, iX2, iY1, iY2};
  sort(reordering.begin(), reordering.end());
  int countCorrect = 0;
  int countIncorrect = 0;
  do {
    // TODO: Precondition (can be broken) bottom most loop must remain unchanged if sparse
    bool valid_reordering = reordering[3] == iY2;

    if (!valid_reordering) {
      continue;
    }

    IndexStmt reordered = stmt.reorder(reordering);
    C.compile(reordered);
    C.assemble();
    C.compute();
    if (!equals(C, expected)) {
      //cout << util::join(reordering) << endl;
      countIncorrect++;

      std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);
      ir::Stmt compute = lower(reordered, "compute",  false, true);
      codegen->compile(compute, true);
      ASSERT_TENSOR_EQ(expected, C);
      exit(1);
    }
    else {
      countCorrect++;
    }
  } while (next_permutation(reordering.begin(), reordering.end()));
}

TEST(scheduling, pos_noop) {
  Tensor<double> A("A", {8}, {Sparse});
  Tensor<double> C("C");

  for (int i = 0; i < 8; i++) {
    if (i % 2 == 0) {
      A.insert({i}, (double) i);
    }
  }

  A.pack();

  IndexVar i("i"), ipos("ipos");
  C = A(i);

  IndexStmt stmt = C.getAssignment().concretize();
  stmt = stmt.pos(i, ipos, A(i));

//  ir::CodeGen_C codegen = ir::CodeGen_C(cout, ir::CodeGen::ImplementationGen, false);
//  ir::Stmt compute = lower(stmt, "compute",  false, true);
//  codegen.print(compute);

  C.compile(stmt);
  C.assemble();
  C.compute();

  Tensor<double> expected("expected");
  expected = A(i);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, C);
}

TEST(scheduling, pos_mul_dense) {
  Tensor<double> A("A", {8}, {Sparse});
  Tensor<double> B("B", {8}, {Dense});
  Tensor<double> C("C", {8}, {Dense});

  for (int i = 0; i < 8; i++) {
    if (i % 2 == 0) {
      A.insert({i}, (double) i);
    }
    B.insert({i}, (double) i);
  }

  A.pack();
  B.pack();

  IndexVar i("i"), ipos("ipos");
  C(i) = A(i) * B(i);

  IndexStmt stmt = C.getAssignment().concretize();
  stmt = stmt.pos(i, ipos, A(i));

//  ir::CodeGen_C codegen = ir::CodeGen_C(cout, ir::CodeGen::ImplementationGen, true);
//  ir::Stmt compute = lower(stmt, "compute",  false, true);
//  codegen.print(compute);

  C.compile(stmt);
  C.assemble();
  C.compute();

  Tensor<double> expected("expected", {8}, {Dense});
  expected(i) = A(i) * B(i);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, C);
}

TEST(scheduling, pos_mul_sparse) {
  Tensor<double> A("A", {8}, {Sparse});
  Tensor<double> B("B", {8}, {Sparse});
  Tensor<double> C("C", {8}, {Dense});

  for (int i = 0; i < 8; i++) {
    if (i % 2 == 0) {
      A.insert({i}, (double) i);
    }
    if (i != 2 && i != 3 && i != 4) {
      B.insert({i}, (double) i);
    }
  }

  A.pack();
  B.pack();

  IndexVar i("i"), ipos("ipos");
  C(i) = A(i) * B(i);

  IndexStmt stmt = C.getAssignment().concretize();
  stmt = stmt.pos(i, ipos, A(i));

//  ir::CodeGen_C codegen = ir::CodeGen_C(cout, ir::CodeGen::ImplementationGen, false);
//  ir::Stmt compute = lower(stmt, "compute",  false, true);
//  codegen.print(compute);

  C.compile(stmt);
  C.assemble();
  C.compute();

  Tensor<double> expected("expected", {8}, {Dense});
  expected(i) = A(i) * B(i);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, C);
}

TEST(scheduling, pos_mul_dense_split) {
  Tensor<double> A("A", {8}, {Sparse});
  Tensor<double> B("B", {8}, {Dense});
  Tensor<double> C("C", {8}, {Dense});

  for (int i = 0; i < 8; i++) {
    if (i % 2 == 0) {
      A.insert({i}, (double) i);
    }
    B.insert({i}, (double) i);
  }

  A.pack();
  B.pack();

  IndexVar i("i"), ipos("ipos"), iposOuter("iposOuter"), iposInner("iposInner");
  C(i) = A(i) * B(i);

  IndexStmt stmt = C.getAssignment().concretize();
  stmt = stmt.pos(i, ipos, A(i)).split(ipos, iposOuter, iposInner, 2);

//  ir::CodeGen_C codegen = ir::CodeGen_C(cout, ir::CodeGen::ImplementationGen, true);
//  ir::Stmt compute = lower(stmt, "compute",  false, true);
//  codegen.print(compute);

  C.compile(stmt);
  C.assemble();
  C.compute();

  Tensor<double> expected("expected", {8}, {Dense});
  expected(i) = A(i) * B(i);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, C);
}

TEST(scheduling, pos_tile_coord_and_pos) {
  Tensor<double> A("A", {8}, {Sparse});
  Tensor<double> B("B", {8}, {Dense});
  Tensor<double> C("C", {8}, {Dense});

  for (int i = 0; i < 8; i++) {
    if (i % 2 == 0) {
      A.insert({i}, (double) i);
    }
    B.insert({i}, (double) i);
  }

  A.pack();
  B.pack();

  IndexVar i("i"), iOuter("iOuter"), iInner("iInner"), ipos("ipos"), iposOuter("iposOuter"), iposInner("iposInner");
  C(i) = A(i) * B(i);

  IndexStmt stmt = C.getAssignment().concretize();
  stmt = stmt.split(i, iOuter, iInner, 4)
          .pos(iInner, ipos, A(i)).split(ipos, iposOuter, iposInner, 2);

//  ir::CodeGen_C codegen = ir::CodeGen_C(cout, ir::CodeGen::ImplementationGen, true);
//  ir::Stmt compute = lower(stmt, "compute",  false, true);
//  codegen.print(compute);

  C.compile(stmt);
  C.assemble();
  C.compute();

  Tensor<double> expected("expected", {8}, {Dense});
  expected(i) = A(i) * B(i);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, C);
}

TEST(scheduling, spmv_warp_per_row) {
  if (!should_use_CUDA_codegen()) {
    return;
  }
  const int WARP_SIZE = 32;
  const int BLOCK_SIZE = 256;
  const int ROWS_PER_WARP = 4;
  const int WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;
  const int ROWS_PER_BLOCK = ROWS_PER_WARP * WARPS_PER_BLOCK;

  const int iSIZE = 1024;
  const int jSIZE = 1024;
  Tensor<double> A("A", {iSIZE, jSIZE}, CSR);
  Tensor<double> x("x", {jSIZE}, {Dense});
  Tensor<double> y("y", {iSIZE}, {Dense});

  for (int i = 0; i < iSIZE; i++) {
    for (int j = 0; j < jSIZE; j++) {
      if ((i+j) % 2 == 0) {
        A.insert({i, j}, (double) 1);
      }
    }
    x.insert({i}, (double) (i));
  }

  A.pack();
  x.pack();

  IndexVar i("i"), j("j"), jpos("jpos");
  IndexVar block("block"), warp("warp"), thread("thread"), warp_row("warp_row"), thread_element("thread_element");
  IndexVar block_row("block_row");

  y(i) = A(i, j) * x(j);

  IndexStmt stmt = y.getAssignment().concretize();
  stmt = stmt.split(i, block, block_row, ROWS_PER_BLOCK)
          .split(block_row, warp_row, warp, WARPS_PER_BLOCK)
          .pos(j, jpos, A(i, j))
          .split(jpos, thread_element, thread, WARP_SIZE)
          .reorder({block, warp, warp_row, thread, thread_element})
          .parallelize(block, ParallelUnit::GPUBlock, OutputRaceStrategy::IgnoreRaces)
          .parallelize(warp, ParallelUnit::GPUWarp, OutputRaceStrategy::IgnoreRaces)
          .parallelize(thread, ParallelUnit::GPUThread, OutputRaceStrategy::Temporary);
//  ir::CodeGen_CUDA codegen = ir::CodeGen_CUDA(cout, ir::CodeGen_CUDA::ImplementationGen);
//  ir::Stmt compute = lower(stmt, "compute",  false, true);
//  codegen.print(compute);

  y.compile(stmt);
  y.assemble();
  y.compute();

  Tensor<double> expected("expected", {iSIZE}, {Dense});
  expected(i) = A(i, j) * x(j);
  stmt = expected.getAssignment().concretize();
  expected.compile(stmt);
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, y);
}

TEST(scheduling, dense_pos_error) {
  Tensor<double> x("x", {8}, {Dense});
  Tensor<double> y("y", {8}, {Dense});
  IndexVar i("i"), ipos("ipos");
  y(i) = x(i);

  IndexStmt stmt = y.getAssignment().concretize();
#ifdef PYTHON
  ASSERT_THROW(stmt.pos(i, ipos, x(i)), taco::TacoException);
#else
  ASSERT_DEATH(stmt.pos(i, ipos, x(i)), "Pos transformation is not valid for dense formats, the coordinate space should be transformed instead");
#endif
}

TEST(scheduling, pos_var_not_in_access) {
  Tensor<double> x("x", {8}, {Dense});
  Tensor<double> y("y", {8}, {Dense});
  IndexVar i("i"), ipos("ipos"), j("j");
  y(i) = x(i);

  IndexStmt stmt = y.getAssignment().concretize();
#ifdef PYTHON
  ASSERT_THROW(stmt.pos(j, ipos, x(i)), taco::TacoException);
#else
  ASSERT_DEATH(stmt.pos(j, ipos, x(i)), "Index variable j does not appear in access: x[(]i[)]");
#endif
}

TEST(scheduling, pos_wrong_access) {
  Tensor<double> x("x", {8}, {Dense});
  Tensor<double> y("y", {8}, {Dense});
  IndexVar i("i"), ipos("ipos"), j("j");
  y(i) = x(i);

  IndexStmt stmt = y.getAssignment().concretize();
#ifdef PYTHON
  ASSERT_THROW(stmt.pos(i, ipos, x(j)), taco::TacoException);
  ASSERT_THROW(stmt.pos(i, ipos, y(i)), taco::TacoException);
#else
  ASSERT_DEATH(stmt.pos(i, ipos, x(j)), "Access: x[(]j[)] does not appear in index statement as an argument");
  ASSERT_DEATH(stmt.pos(i, ipos, y(i)), "Access: y[(]i[)] does not appear in index statement as an argument");
#endif
}

TEST(scheduling_eval_test, spmv_fuse) {
  if (!should_use_CUDA_codegen()) return;
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

  srand(59393);
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
    x.insert({j}, (double) ((int) (rand_float*3)));
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
          .parallelize(block, ParallelUnit::GPUBlock, OutputRaceStrategy::IgnoreRaces)
          .parallelize(warp, ParallelUnit::GPUWarp, OutputRaceStrategy::Atomics)
          .parallelize(thread, ParallelUnit::GPUThread, OutputRaceStrategy::Atomics);
//  ir::CodeGen_CUDA codegen = ir::CodeGen_CUDA(cout, ir::CodeGen_CUDA::ImplementationGen);
//  ir::Stmt compute = lower(stmt, "compute",  false, true);
//  codegen.print(compute);

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
