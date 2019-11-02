#include <taco/index_notation/transformations.h>
#include <codegen/codegen_c.h>
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

  Tensor<double> expected({4, 4}, {Dense, Dense});
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
          .parallelize(i0, should_use_CUDA_codegen() ? PARALLEL_UNIT::GPU_BLOCK : PARALLEL_UNIT::CPU_THREAD, OUTPUT_RACE_STRATEGY::ATOMICS);

  if (should_use_CUDA_codegen()) {
    stmt = stmt.parallelize(j0, PARALLEL_UNIT::GPU_THREAD, OUTPUT_RACE_STRATEGY::ATOMICS);
  }

  C.compile(stmt);
  C.assemble();
  C.compute();

  Tensor<double> expected({8, 8}, {Dense, Dense});
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
            .parallelize(block, PARALLEL_UNIT::GPU_BLOCK, OUTPUT_RACE_STRATEGY::ATOMICS)
            .parallelize(thread, PARALLEL_UNIT::GPU_THREAD, OUTPUT_RACE_STRATEGY::ATOMICS);
  }
  else {
    stmt = stmt.split(i, i0, i1, 2)
            .parallelize(i0, PARALLEL_UNIT::CPU_THREAD, OUTPUT_RACE_STRATEGY::ATOMICS);
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
            .parallelize(block, PARALLEL_UNIT::GPU_BLOCK, OUTPUT_RACE_STRATEGY::TEMPORARY)
            .parallelize(thread, PARALLEL_UNIT::GPU_THREAD, OUTPUT_RACE_STRATEGY::TEMPORARY);
  }
  else {
    stmt = stmt.split(i, i0, i1, 2)
            .parallelize(i0, PARALLEL_UNIT::CPU_THREAD, OUTPUT_RACE_STRATEGY::TEMPORARY);
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

  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);
  ir::Stmt compute = lower(stmt, "compute",  false, true);
  codegen->compile(compute, true);
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
      cout << util::join(reordering) << endl;
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

  ir::CodeGen_C codegen = ir::CodeGen_C(cout, ir::CodeGen::ImplementationGen, false);
  ir::Stmt compute = lower(stmt, "compute",  false, true);
  codegen.print(compute);

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