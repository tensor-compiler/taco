#include <taco/index_notation/transformations.h>
#include <codegen/codegen_c.h>
#include <codegen/codegen_cuda.h>
#include <codegen/codegen_spatial.h>
#include "test.h"
#include "test_tensors.h"
#include "taco/tensor.h"
#include "taco/index_notation/index_notation.h"
#include "codegen/codegen.h"
#include "taco/lower/lower.h"
#include "taco/spatial.h"
#include "taco/cuda.h"

using namespace taco;
const IndexVar i("i"), j("j"), k("k");

TEST(spatial, vecElemMul) {
  // Enable spatial codegen
  //should_use_Spatial_codegen();

  Tensor<double> A("A", {16}, {Dense});
  Tensor<double> B("B", {16}, {Dense});
  Tensor<double> C("C", {16}, {Dense});

  for (int i = 0; i < 16; i++) {
      C.insert({i}, (double) i);
      B.insert({i}, (double) i);
  }

  IndexVar i("i");
  IndexVar i0("i0"), i1("i1");
  A(i) = B(i) * C(i);

  IndexStmt stmt = A.getAssignment().concretize();
  A.compile(stmt);
  A.assemble();
  A.compute();

  Tensor<double> expected("expected", {16}, {Dense});
  expected(i) = B(i) * C(i);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(A, expected);

  set_Spatial_codegen_enabled(true);

  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);
  ir::Stmt compute = lower(stmt, "compute",  false, true);
  codegen->compile(compute, false);
  set_Spatial_codegen_enabled(false);
}

TEST(spatial, tileCompute_vecElemMul) {
  set_Spatial_codegen_enabled(false);

  Tensor<double> A("A", {16}, {Dense});
  Tensor<double> B("B", {16}, {Dense});
  Tensor<double> C("C", {16}, {Dense});

  for (int i = 0; i < 16; i++) {
      A.insert({i}, (double) i);
      B.insert({i}, (double) i);
  }

  B.pack();
  C.pack();

  IndexVar i("i");
  IndexVar i_bounded("i_bounded");
  IndexVar i0("i0"), i1("i1");
  IndexExpr precomputedExpr = B(i) * C(i);
  A(i) = precomputedExpr;

  IndexStmt stmt = A.getAssignment().concretize();
  TensorVar precomputed("precomputed", Type(Float64, {Dimension(i1)}), taco::dense);
  stmt = stmt.bound(i, i_bounded, 16, BoundType::MaxExact)
             .split(i_bounded, i0, i1, 4)
             .precompute(precomputedExpr, i1, i1, precomputed);

  ir::IRPrinter irp = ir::IRPrinter(cout);
  cout << "----------------Post-Schedule Stmt-----------------" << endl;
  cout << stmt << endl;

  A.compile(stmt);
  A.assemble();
  A.compute();

  Tensor<double> expected("expected", {16}, {Dense});
  expected(i) = B(i) * C(i);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(A, expected);

  set_Spatial_codegen_enabled(true);

  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);
  ir::Stmt compute = lower(stmt, "compute",  false, true);
  codegen->compile(compute, false);
  set_Spatial_codegen_enabled(false);
}

TEST(spatial, tile_matElemMul) {
  set_Spatial_codegen_enabled(false);

  Tensor<double> A("A", {16, 16}, {Dense, Dense});
  Tensor<double> B("B", {16, 16}, {Dense, Dense});
  Tensor<double> C("C", {16, 16}, {Dense, Dense});

  for (int i = 0; i < 16; i++) {
    for (int j = 0; j < 16; j++) {
      C.insert({i, j}, (double) i+j);
      B.insert({i, j}, (double) i+j);
    }
  }

  B.pack();
  C.pack();

  IndexVar i("i");
  IndexVar i_bounded("i_bounded");
  IndexVar i0("i0"), i1("i1"), i0_bounded("i0_bounded"), i0_bounded1("i0_bounded1");

  IndexVar j("j"), j_bounded("j_bounded");
  IndexVar j0("j0"), j1("j1"), j0_bounded("j0_bounded"), j0_bounded1("j0_bounded1");

  IndexExpr precomputedExpr = B(i, j) * C(i, j);
  A(i, j) = precomputedExpr;

  IndexStmt stmt = A.getAssignment().concretize();
  TensorVar precomputed("precomputed", Type(Float64, {Dimension(i1), Dimension(j1)}), taco::dense);
  stmt = stmt.bound(i, i_bounded, 16, BoundType::MaxExact)
          .split(i_bounded, i0, i1, 4)
          .bound(j, j_bounded, 16, BoundType::MaxExact)
          .split(j_bounded, j0, j1, 4);

  ir::IRPrinter irp = ir::IRPrinter(cout);
  cout << stmt << endl;

  A.compile(stmt);
  A.assemble();
  A.compute();

  Tensor<double> expected("expected", {16, 16}, {Dense});
  expected(i, j) = B(i, j) * C(i, j);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(A, expected);

  set_Spatial_codegen_enabled(true);
//  ir::IRPrinter irp = ir::IRPrinter(cout);
//  cout << stmt << endl;
//
  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);
  ir::Stmt compute = lower(stmt, "Compute",  false, true);
//
//  irp.print(compute);
//  cout << endl;
  codegen->compile(compute, false);
  set_Spatial_codegen_enabled(false);
}

TEST(spatial, reduction_dotProduct) {
  set_Spatial_codegen_enabled(false);

  Tensor<int> A("A");
  Tensor<int> B("B", {16}, {Dense});
  Tensor<int> C("C", {16}, {Dense});

  for (int i = 0; i < 16; i++) {
    C.insert({i}, (int) i);
    B.insert({i}, (int) i);
  }

  B.pack();
  C.pack();

  IndexVar i("i"), i_b("i_b");
  A() = B(i) * C(i);

  IndexStmt stmt = A.getAssignment().concretize();
  stmt = stmt.bound(i, i_b, 16, BoundType::MaxExact)
             .parallelize(i_b, ParallelUnit::Spatial,
                          OutputRaceStrategy::SpatialReduction);

  ir::IRPrinter irp = ir::IRPrinter(cout);
  cout << stmt << endl;

  set_Spatial_codegen_enabled(true);

  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);
  ir::Stmt compute = lower(stmt, "Compute",  false, true);

  cout << "----------Finish codegen lowering---------" << endl;
  cout << compute << endl;

  cout << "-----------Spatial Code---------------" << endl;
  codegen->compile(compute, false);
  set_Spatial_codegen_enabled(false);
}

TEST(spatial, bound_elemMul) {
  set_Spatial_codegen_enabled(false);

  Tensor<double> A("A", {16}, {Dense});
  Tensor<double> B("B", {16}, {Dense});
  Tensor<double> C("C", {16}, {Dense});

  for (int i = 0; i < 16; i++) {
    C.insert({i}, (double) i);
    B.insert({i}, (double) i);
  }

  B.pack();
  C.pack();

  IndexVar i("i"), i_b("i_b");


  A(i) = B(i) * C(i);

  IndexStmt stmt = A.getAssignment().concretize();
  stmt = stmt.bound(i, i_b, 16, BoundType::MaxExact);
  //.parallelize(i, ParallelUnit::Spatial, OutputRaceStrategy::SpatialReduction);

  ir::IRPrinter irp = ir::IRPrinter(cout);
  cout << stmt << endl;

  A.compile(stmt);
  A.assemble();
  A.compute();

  Tensor<double> expected("expected", {16});
  expected(i) = B(i) * C(i);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(A, expected);

  cout << "----------------Resulting Tensors-----------------" << endl;
  cout << A << endl;
  cout << expected << endl;

  set_Spatial_codegen_enabled(true);
  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);
  ir::Stmt compute = lower(stmt, "Compute", false, true);
  cout << "----------Finish codegen lowering---------" << endl;
  cout << compute << endl;

  cout << "-----------Spatial Code---------------" << endl;
  codegen->compile(compute, false);
  set_Spatial_codegen_enabled(false);
}

TEST(spatial, reduction_GEMV) {
  set_Spatial_codegen_enabled(false);

  Tensor<int> A("A", {16}, {Dense});
  Tensor<int> B("B", {16, 16}, {Dense, Dense});
  Tensor<int> C("C", {16}, {Dense});


  for (int i = 0; i < 16; i++) {
    for (int j = 0; j < 16; j++) {
      B.insert({i, j}, (int) i + j);
    }
    C.insert({i}, (int) i);
  }

  B.pack();
  C.pack();

  IndexVar i("i"), i_b("i_b");
  IndexVar j("j"), j_b("j_b");

  A(i) = B(i, j) * C(j);

  IndexStmt stmt = A.getAssignment().concretize();
  stmt = scalarPromote(stmt);
  stmt = stmt
          .bound(i, i_b, 16, BoundType::MaxExact)
          .bound(j, j_b, 16, BoundType::MaxExact)
          .parallelize(j_b, ParallelUnit::Spatial, OutputRaceStrategy::SpatialReduction);



  ir::IRPrinter irp = ir::IRPrinter(cout);
  cout << "Concrete Statement: " << stmt << endl;

  A.compile(stmt);
  A.assemble();
  A.compute();

//  Tensor<int> expected("expected", {16}, {Dense});
//  expected(i) = B(i, j) * C(j);
//  expected.compile();
//  expected.assemble();
//  expected.compute();
//  ASSERT_TENSOR_EQ(A, expected);
//
//  cout << "----------------Resulting Tensors-----------------" << endl;
//  cout << A << endl;
//  cout << expected << endl;

  set_Spatial_codegen_enabled(true);
  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);
  ir::Stmt compute = lower(stmt, "Compute", false, true);
  cout << "----------Finish codegen lowering---------" << endl;
  cout << compute << endl;

  cout << "-----------Spatial Code---------------" << endl;
  codegen->compile(compute, false);
  set_Spatial_codegen_enabled(false);
}

TEST(spatial, reduction_GEMM) {
  set_Spatial_codegen_enabled(false);

  Tensor<int> A("A", {16, 16}, {Dense, Dense});
  Tensor<int> B("B", {16, 16}, {Dense, Dense});
  Tensor<int> C("C", {16, 16}, {Dense, Dense});


  for (int i = 0; i < 16; i++) {
    for (int j = 0; j < 16; j++) {
      B.insert({i, j}, (int) i + j);
      C.insert({i, j}, (int) i+j);
    }
  }

  B.pack();
  C.pack();

  IndexVar i("i"), i_b("i_b");
  IndexVar j("j"), j_b("j_b");
  IndexVar k("k"), k_b("k_b");

  A(i, j) = B(i, k) * C(k, j);

  // [Spatial] TODO: Add in temporary workspace
  IndexStmt stmt = A.getAssignment().concretize();
  stmt = scalarPromote(stmt);
  stmt = stmt
          .bound(i, i_b, 16, BoundType::MaxExact)
          .bound(j, j_b, 16, BoundType::MaxExact)
          .bound(k, k_b, 16, BoundType::MaxExact)
          .parallelize(k_b, ParallelUnit::Spatial, OutputRaceStrategy::SpatialReduction, 16);
  ir::IRPrinter irp = ir::IRPrinter(cout);
  cout << stmt << endl;

  cout << stmt << endl;

  A.compile(stmt);
  A.assemble();
  A.compute();

  Tensor<int> expected("expected", {16, 16}, {Dense, Dense});
  expected(i, j) = B(i, k) * C(k, j);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(A, expected);

  cout << "----------------Resulting Tensors-----------------" << endl;
  cout << A << endl;
  cout << expected << endl;

  set_Spatial_codegen_enabled(true);
  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);
  ir::Stmt compute = lower(stmt, "Compute", false, true);
  cout << "----------Finish codegen lowering---------" << endl;
  cout << compute << endl;

  cout << "-----------Spatial Code---------------" << endl;
  codegen->compile(compute, false);
  set_Spatial_codegen_enabled(false);
}

TEST(spatial, outerProduct) {
  set_Spatial_codegen_enabled(false);

  Tensor<int> A("A", {16, 16}, {Dense, Dense});
  Tensor<int> B("B", {16}, {Dense});
  Tensor<int> C("C", {16}, {Dense});


  for (int i = 0; i < 16; i++) {
    B.insert({i}, (int) i);
    C.insert({i}, (int) i);
  }

  B.pack();
  C.pack();

  IndexVar i("i"), i_b("i_b");
  IndexVar j("j"), j_b("j_b");

  A(i, j) = B(i) * C(j);

  // [Spatial] TODO: Add in temporary workspace
  IndexStmt stmt = A.getAssignment().concretize();
  stmt = scalarPromote(stmt);
  stmt = stmt
          .bound(i, i_b, 16, BoundType::MaxExact)
          .bound(j, j_b, 16, BoundType::MaxExact);



  ir::IRPrinter irp = ir::IRPrinter(cout);
  cout << stmt << endl;

  A.compile(stmt);
  A.assemble();
  A.compute();

  Tensor<int> expected("expected", {16, 16}, {Dense, Dense});
  expected(i, j) = B(i) * C(j);;
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(A, expected);

  cout << "----------------Resulting Tensors-----------------" << endl;
  cout << A << endl;
  cout << expected << endl;

  set_Spatial_codegen_enabled(true);
  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);
  ir::Stmt compute = lower(stmt, "Compute", false, true);
  cout << "----------Finish codegen lowering---------" << endl;
  cout << compute << endl;

  cout << "-----------Spatial Code---------------" << endl;
  codegen->compile(compute, false);
  set_Spatial_codegen_enabled(false);
}

TEST(spatial, reduction_higherOrder) {
  set_Spatial_codegen_enabled(false);

  Tensor<int> A("A", {16, 16, 16}, {Dense, Dense, Dense});
  Tensor<int> B("B", {16, 16, 16}, {Dense, Dense, Dense});
  Tensor<int> C("C", {16, 16}, {Dense, Dense});


  for (int i = 0; i < 16; i++) {
    for (int j = 0; j < 16; j++) {
      for (int k = 0; k < 16; k++) {
        B.insert({i, j, k}, (int) i+j+k);

      }
      C.insert({i, j}, (int) i+j);
    }
  }

  B.pack();
  C.pack();

  IndexVar i("i"), i_b("i_b");
  IndexVar j("j"), j_b("j_b");
  IndexVar k("j"), k_b("j_b");
  IndexVar l("j"), l_b("j_b");

  A(i, j, l) = B(i, j, k) * C(k, l);

  // [Spatial] TODO: Add in temporary workspace
  IndexStmt stmt = A.getAssignment().concretize();
  stmt = scalarPromote(stmt);
  stmt = stmt
          .bound(i, i_b, 16, BoundType::MaxExact)
          .bound(j, j_b, 16, BoundType::MaxExact)
          .bound(k, k_b, 16, BoundType::MaxExact)
          .bound(l, l_b, 16, BoundType::MaxExact)
          .parallelize(k_b, ParallelUnit::Spatial, OutputRaceStrategy::SpatialReduction);

  ir::IRPrinter irp = ir::IRPrinter(cout);
  cout << stmt << endl;

  A.compile(stmt);
  A.assemble();
  A.compute();

  Tensor<int> expected("expected", {16, 16, 16}, {Dense, Dense, Dense});
  expected(i, j, l) = B(i, j, k) * C(k, l);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(A, expected);

  cout << "----------------Resulting Tensors-----------------" << endl;
  cout << A << endl;
  cout << expected << endl;

  set_Spatial_codegen_enabled(true);
  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);
  ir::Stmt compute = lower(stmt, "Compute", false, true);
  cout << "----------Finish codegen lowering---------" << endl;
  cout << compute << endl;

  cout << "-----------Spatial Code---------------" << endl;
  codegen->compile(compute, false);
}

TEST(spatial, tile_vecElemMul) {
  set_Spatial_codegen_enabled(false);

  int n = 1024;
  Tensor<double> A("A", {n}, {Dense}, MemoryLocation::SpatialDRAM);
  Tensor<double> B("B", {n}, {Dense}, MemoryLocation::SpatialDRAM);
  Tensor<double> C("C", {n}, {Dense}, MemoryLocation::SpatialDRAM);

  for (int i = 0; i < n; i++) {
    B.insert({i}, (double) i);
    C.insert({i}, (double) i);
  }

  B.pack();
  C.pack();

  IndexVar i("i");
  IndexVar i_bounded("i_bounded");
  IndexVar i0("i0"), i1("i1");
  IndexVar i2("i2");
  IndexExpr BExpr = B(i);
  IndexExpr CExpr = C(i);
  IndexExpr precomputedExpr = (BExpr) * (CExpr);
  A(i) = precomputedExpr;

  IndexStmt stmt = A.getAssignment().concretize();
  TensorVar B_sram("B_sram", Type(Float64, {Dimension(i1)}), taco::dense, MemoryLocation::SpatialSRAM);
  TensorVar C_sram("C_sram", Type(Float64, {Dimension(i1)}), taco::dense, MemoryLocation::SpatialSRAM);
  TensorVar precomputed("precomputed", Type(Float64, {Dimension(i1)}), taco::dense, MemoryLocation::SpatialSRAM);

  ir::IRPrinter irp = ir::IRPrinter(cout);
  cout << "----------------Pre-Schedule Stmt-----------------" << endl;
  cout << stmt << endl;

  stmt = stmt.bound(i, i_bounded, n, BoundType::MaxExact)
          .split(i_bounded, i0, i1, 16)
          .parallelize(i0, ParallelUnit::Spatial, OutputRaceStrategy::IgnoreRaces, 2)
          .precompute(precomputedExpr, i1, i1, precomputed);
  cout << "----------------Post-Schedule 1 Stmt-----------------" << endl;
  cout << stmt << endl;

  stmt = stmt.precompute(BExpr, i1, i1, B_sram); // where (forall(i p = B_sram * C_sram, forall_i (B_sram = B(i))
  cout << "----------------Post-Schedule 2 Stmt-----------------" << endl;
  cout << stmt << endl;

  stmt = stmt.precompute(CExpr, i1, i1, C_sram)
          .parallelize(i1, ParallelUnit::Spatial, OutputRaceStrategy::IgnoreRaces, 16);


  cout << "----------------Post-Schedule 3 Stmt-----------------" << endl;
  cout << stmt << endl;

  set_Spatial_codegen_enabled(true);

  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);
  ir::Stmt compute = lower(stmt, "compute",  false, true);

  cout << "----------Finish codegen lowering---------" << endl;
  cout << compute << endl;

  codegen->compile(compute, false);
  set_Spatial_codegen_enabled(false);
}

TEST(spatial, tile_dotProduct) {
  set_Spatial_codegen_enabled(false);

  int n = 1024;
  Tensor<int> A("A", {}, {}, MemoryLocation::SpatialReg);
  Tensor<int> B("B", {n}, {Dense}, MemoryLocation::SpatialDRAM);
  Tensor<int> C("C", {n}, {Dense}, MemoryLocation::SpatialDRAM);

  for (int i = 0; i < n; i++) {
    B.insert({i}, (int) i);
    C.insert({i}, (int) i);
  }

  B.pack();
  C.pack();

  IndexVar i("i");
  IndexVar i_bounded("i_bounded");
  IndexVar i0("i0"), i1("i1");
  IndexExpr BExpr = B(i);
  IndexExpr CExpr = C(i);
  IndexExpr precomputedExpr = (BExpr) * (CExpr);
  A() = precomputedExpr;

  IndexStmt stmt = A.getAssignment().concretize();
  TensorVar B_sram("B_sram", Type(Float64, {Dimension(i1)}), taco::dense, MemoryLocation::SpatialSRAM);
  TensorVar C_sram("C_sram", Type(Float64, {Dimension(i1)}), taco::dense, MemoryLocation::SpatialSRAM);
  TensorVar precomputed("precomputed", Type(Float64, {Dimension(i1)}), taco::dense, MemoryLocation::SpatialSRAM);

  ir::IRPrinter irp = ir::IRPrinter(cout);
  cout << "----------------Pre-Schedule Stmt-----------------" << endl;
  cout << stmt << endl;

  stmt = stmt.bound(i, i_bounded, n, BoundType::MaxExact)
          .split(i_bounded, i0, i1, 32)
          .precompute(precomputedExpr, i0, i0, precomputed)
          .parallelize(i0, ParallelUnit::Spatial, OutputRaceStrategy::SpatialReduction, 2);
  cout << "----------------Post-Schedule 1 Stmt-----------------" << endl;
  cout << stmt << endl;

  stmt = stmt.precompute(BExpr, i1, i1, B_sram);
  cout << "----------------Post-Schedule 2 Stmt-----------------" << endl;
  cout << stmt << endl;

  stmt = stmt.precompute(CExpr, i1, i1, C_sram)
          .parallelize(i1, ParallelUnit::Spatial, OutputRaceStrategy::SpatialReduction, 1);


  cout << "----------------Post-Schedule 3 Stmt-----------------" << endl;
  cout << stmt << endl;

  set_Spatial_codegen_enabled(true);

  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);
  ir::Stmt compute = lower(stmt, "compute",  false, true);

  cout << "----------Finish codegen lowering---------" << endl;
  cout << compute << endl;

  codegen->compile(compute, false);
  set_Spatial_codegen_enabled(false);
}

TEST(spatial, tile_GEMV) {
  set_Spatial_codegen_enabled(false);

  int n = 16;
  Tensor<int> A("A", {n}, {Dense}, MemoryLocation::SpatialDRAM);
  Tensor<int> B("B", {n, n}, {Dense, Dense}, MemoryLocation::SpatialDRAM);
  Tensor<int> C("C", {n}, {Dense}, MemoryLocation::SpatialDRAM);

  for (int j = 0; j < n; j++) {
    for (int i = 0; i < n; i++) {
      B.insert({i, j}, (int) j*n+i);
      C.insert({i}, (int) i);
    }
  }

  B.pack();
  C.pack();

  IndexVar i("i"), j("j");
  IndexVar i_bounded("i_bounded"), j_bounded("j_bounded");
  IndexVar i0("i0"), i1("i1");
  IndexVar j0("j0"), j1("j1");
  IndexExpr BExpr = B(i, j);
  IndexExpr CExpr = C(j);
  IndexExpr precomputedExpr = (BExpr) * (CExpr);
  A(i) = precomputedExpr;

  IndexStmt stmt = A.getAssignment().concretize();
  //TensorVar B_sram("B_sram", Type(Int64 , {Dimension(i1), Dimension(j1)}), taco::dense, MemoryLocation::SpatialSRAM);
  TensorVar C_sram("C_sram", Type(Int64, {Dimension(j1)}), taco::dense, MemoryLocation::SpatialSRAM);
  TensorVar precomputed("precomputed", Type(Float64, {Dimension(j1)}), taco::dense, MemoryLocation::SpatialSRAM);
  TensorVar A_sram("A_sram", Type(Float64, {Dimension(i1)}), taco::dense, MemoryLocation::SpatialSRAM);

  ir::IRPrinter irp = ir::IRPrinter(cout);
  cout << "----------------Pre-Schedule Stmt-----------------" << endl;
  cout << stmt << endl;

  stmt = stmt.bound(i, i_bounded, n, BoundType::MaxExact)
          .bound(j, j_bounded, n, BoundType::MaxExact)
          .split(i_bounded, i0, i1, 4)
          .split(j_bounded, j0, j1, 4)
          .parallelize(i0, ParallelUnit::Spatial, OutputRaceStrategy::SpatialReduction, 1)
          .parallelize(j0, ParallelUnit::Spatial, OutputRaceStrategy::SpatialReduction, 1)
          .precompute(precomputedExpr, j1, j1, precomputed);
  cout << "----------------Post-Schedule 1 Stmt-----------------" << endl;
  //stmt = scalarPromote(stmt);
  cout << stmt << endl;

  cout << "----------------Post-Schedule 2 Stmt-----------------" << endl;
  cout << stmt << endl;

  // TODO: fix this app to use MemReduce


  set_Spatial_codegen_enabled(true);

  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);
  ir::Stmt compute = lower(stmt, "compute",  false, true);

  cout << "----------Finish codegen lowering---------" << endl;
  cout << compute << endl;

  codegen->compile(compute, false);
  set_Spatial_codegen_enabled(false);
}

TEST(spatial, DISABLED_sparse_csr_spMV_fused) {
  set_Spatial_codegen_enabled(false);

  Tensor<double> A("A", {16}, {Sparse}, taco::MemoryLocation::SpatialDRAM);
  Tensor<double> B("B", {16, 16}, CSR, taco::MemoryLocation::SpatialDRAM);
  Tensor<double> C("C", {16}, {Dense}, taco::MemoryLocation::SpatialDRAM);

  for (int i = 0; i < 16; i++) {
    if (i % 4 == 0)
      C.insert({i}, (double) i);
    B.insert({i, i}, (double) i);
  }

  IndexVar i("i"), j("j"), f("f"), fp("fp");
  A(i) = B(i, j) * C(j);

  IndexStmt stmt = A.getAssignment().concretize();
  stmt = stmt.fuse(i, j, f).pos(f, fp, B(i,j));
  stmt = stmt.parallelize(j, ParallelUnit::Spatial, OutputRaceStrategy::SpatialReduction);
  ir::IRPrinter irp = ir::IRPrinter(cout);

  set_Spatial_codegen_enabled(true);

  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);
  ir::Stmt compute = lower(stmt, "compute",  false, true);
  codegen->compile(compute, false);
  set_Spatial_codegen_enabled(false);
}

TEST(spatial, sparse_csr_spMV_default) {
  set_Spatial_codegen_enabled(false);
  int N = 32;
  Tensor<int> a("a", {N}, {Dense}, taco::MemoryLocation::SpatialFIFO);
  Tensor<int> B("B", {N, N}, CSR, taco::MemoryLocation::SpatialFIFORetimed);
  Tensor<int> c("c", {N}, {Dense}, taco::MemoryLocation::SpatialSparseSRAM);

  for (int i = 0; i < N; i++) {
    if (i % 4 == 0)
      c.insert({i}, (int) i);
    B.insert({i, i}, (int) i);
  }

  IndexVar i("i"), j("j"), f("f"), fp("fp");
  IndexVar ib("ib"), jb("jb");
  a(i) = B(i, j) * c(j);

  IndexStmt stmt = a.getAssignment().concretize();
  stmt = stmt.parallelize(j, ParallelUnit::Spatial, OutputRaceStrategy::SpatialReduction, 16);
  stmt = scalarPromote(stmt);

  cout << "----------------Post-Schedule Stmt-----------------" << endl;
  cout << stmt << endl;

  ir::IRPrinter irp = ir::IRPrinter(cout);

  cout << "----------------CPU LLIR-----------------" << endl;
  ir::Stmt compute = lower(stmt, "compute",  false, true);
  irp.print(compute);
  cout << endl;

  cout << "----------------SPATIAL LLIR-----------------" << endl;
  set_Spatial_codegen_enabled(true);
  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);
  ir::Stmt computes = lower(stmt, "SpMV",  false, true);
  irp.print(computes);

  cout << "----------------SPATIAL CODEGEN-----------------" << endl;
  codegen->compile(computes, false);
  set_Spatial_codegen_enabled(false);
}

TEST(spatial, sparse_csr_spMV_op) {
  set_Spatial_codegen_enabled(false);
  int N = 32;
  Tensor<int> a("a", {N}, {Dense}, taco::MemoryLocation::SpatialSRAM);
  Tensor<int> B("B", {N, N}, CSR, taco::MemoryLocation::SpatialFIFORetimed);
  Tensor<int> c("c", {N}, {Dense}, taco::MemoryLocation::SpatialSparseParSRAMSwizzle);

  for (int i = 0; i < N; i++) {
    if (i % 4 == 0)
      c.insert({i}, (int) i);
    B.insert({i, i}, (int) i);
  }

  IndexVar i("i"), j("j"), f("f"), fp("fp");
  IndexVar ib("ib"), jb("jb");
  a(i) = B(i, j) * c(j);

  IndexStmt stmt = a.getAssignment().concretize();
  stmt = stmt.parallelize(j, ParallelUnit::Spatial, OutputRaceStrategy::SpatialReduction, 16);
  stmt = scalarPromote(stmt);
  stmt = stmt.environment("bp", 2);
  cout << stmt << endl;
  stmt = stmt.communicate(B(i, j), j);

  cout << "----------------Post-Schedule Stmt-----------------" << endl;
  cout << stmt << endl;

  ir::IRPrinter irp = ir::IRPrinter(cout);

  cout << "----------------CPU LLIR-----------------" << endl;
  ir::Stmt compute = lower(stmt, "compute",  false, true);
  irp.print(compute);
  cout << endl;

  cout << "----------------SPATIAL LLIR-----------------" << endl;
  set_Spatial_codegen_enabled(true);
  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);
  ir::Stmt computes = lower(stmt, "SpMV_OP",  false, true);
  irp.print(computes);

  cout << "----------------SPATIAL CODEGEN-----------------" << endl;
  codegen->compile(computes, false);
  set_Spatial_codegen_enabled(false);
}

TEST(spatial, sparse_dense_plus2) {
  set_Spatial_codegen_enabled(true);

  Tensor<int> A("A", {16, 16}, {Dense, Dense}, taco::MemoryLocation::SpatialSparseDRAM);
  Tensor<int> B("B", {16, 16}, CSR, taco::MemoryLocation::SpatialSparseDRAM);
  Tensor<int> C("C", {16, 16}, CSR, taco::MemoryLocation::SpatialSparseDRAM);

  for (int i = 0; i < 16; i++) {
    B.insert({i, i}, (int) i);
    C.insert({i, i}, (int) i);
  }

  IndexVar i("i"), j("j");
  A(i, j) = B(i, j) + C(i, j);

  IndexStmt stmt = A.getAssignment().concretize();

  cout << "----------------Post-Schedule Stmt-----------------" << endl;
  cout << stmt << endl;

  ir::IRPrinter irp = ir::IRPrinter(cout);

  cout << "----------------CPU LLIR-----------------" << endl;
  ir::Stmt compute = lower(stmt, "compute",  false, true);
  irp.print(compute);
  cout << endl;

  cout << "----------------SPATIAL LLIR-----------------" << endl;
  set_Spatial_codegen_enabled(true);
  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);
  ir::Stmt computes = lower(stmt, "compute",  true, true);
  irp.print(computes);

  cout << "----------------SPATIAL CODEGEN-----------------" << endl;
  codegen->compile(computes, false);
  set_Spatial_codegen_enabled(false);
}

TEST(spatial, sparse_csr_plus2) {
  set_Spatial_codegen_enabled(true);
  cout << "------SPATIAL CODEGEN------- " << (int)should_use_Spatial_codegen() << endl;
  Tensor<int> A("A", {16, 16}, {Dense, Sparse}, taco::MemoryLocation::SpatialSparseDRAM);

  Tensor<int> B("B", {16, 16}, {Dense, Sparse}, taco::MemoryLocation::SpatialSparseDRAM);

  Tensor<int> C("C", {16, 16}, {Dense, Sparse}, taco::MemoryLocation::SpatialSparseDRAM);
  set_Spatial_codegen_enabled(false);
  for (int i = 0; i < 16; i++) {
    B.insert({i, i}, (int) i);
    C.insert({i, i}, (int) i);
  }

  IndexVar i("i"), j("j");
  A(i, j) = B(i, j) + C(i, j);

  IndexStmt stmt = A.getAssignment().concretize();

  cout << "----------------Post-Schedule Stmt-----------------" << endl;
  cout << stmt << endl;

  ir::IRPrinter irp = ir::IRPrinter(cout);

  cout << "----------------CPU LLIR-----------------" << endl;
  ir::Stmt compute = lower(stmt, "compute",  false, true);
  irp.print(compute);
  cout << endl;

  cout << "----------------SPATIAL LLIR-----------------" << endl;
  set_Spatial_codegen_enabled(true);
  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);
  ir::Stmt computes = lower(stmt, "Plus2CSR",  false, true);
  irp.print(computes);

  cout << "----------------SPATIAL CODEGEN-----------------" << endl;
  codegen->compile(computes, false);
  set_Spatial_codegen_enabled(false);
}

TEST(spatial, sparse_csr_plus2_op) {
  set_Spatial_codegen_enabled(true);
  cout << "------SPATIAL CODEGEN------- " << (int)should_use_Spatial_codegen() << endl;
  Tensor<int> A("A", {16, 16}, {Dense, Sparse}, taco::MemoryLocation::SpatialSparseDRAM);

  Tensor<int> B("B", {16, 16}, {Dense, Sparse}, taco::MemoryLocation::SpatialSparseDRAM);

  Tensor<int> C("C", {16, 16}, {Dense, Sparse}, taco::MemoryLocation::SpatialSparseDRAM);
  set_Spatial_codegen_enabled(false);
  for (int i = 0; i < 16; i++) {
    B.insert({i, i}, (int) i);
    C.insert({i, i}, (int) i);
  }

  IndexVar i("i"), j("j");
  A(i, j) = B(i, j) + C(i, j);

  IndexStmt stmt = A.getAssignment().concretize();
  stmt = stmt.environment("bp", 2);


  cout << "----------------Post-Schedule Stmt-----------------" << endl;
  cout << stmt << endl;

  ir::IRPrinter irp = ir::IRPrinter(cout);

  cout << "----------------CPU LLIR-----------------" << endl;
  ir::Stmt compute = lower(stmt, "compute",  false, true);
  irp.print(compute);
  cout << endl;

  cout << "----------------SPATIAL LLIR-----------------" << endl;
  set_Spatial_codegen_enabled(true);
  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);
  ir::Stmt computes = lower(stmt, "Plus2CSR_OP",  false, true);
  irp.print(computes);

  cout << "----------------SPATIAL CODEGEN-----------------" << endl;
  codegen->compile(computes, false);
  set_Spatial_codegen_enabled(false);
}

TEST(spatial, sparse_csr_plus3) {
  set_Spatial_codegen_enabled(true);

  Tensor<int> A("A", {16, 16}, CSR, taco::MemoryLocation::SpatialSparseDRAM);
  Tensor<int> B("B", {16, 16}, CSR, taco::MemoryLocation::SpatialSparseDRAM);
  Tensor<int> C("C", {16, 16}, CSR, taco::MemoryLocation::SpatialSparseDRAM);
  Tensor<int> D("D", {16, 16}, CSR, taco::MemoryLocation::SpatialSparseDRAM);

  for (int i = 0; i < 16; i++) {
    B.insert({i, i}, (int) i);
    C.insert({i, i}, (int) i);
    D.insert({i, i}, (int) i);
  }

  IndexVar i("i"), j("j");
  auto precomputedExpr = C(i, j) + D(i, j);
  A(i, j) = B(i, j) + precomputedExpr;

  IndexStmt stmt = A.getAssignment().concretize();
  TensorVar ws("ws", Type(Int(), {16, 16}), {taco::dense, taco::compressed}, MemoryLocation::SpatialSparseSRAM);
  stmt = stmt.precompute(precomputedExpr, {i, j}, {i, j}, ws);

  cout << "----------------Post-Schedule Stmt-----------------" << endl;
  cout << stmt << endl;

  ir::IRPrinter irp = ir::IRPrinter(cout);

  cout << "----------------CPU LLIR-----------------" << endl;
  ir::Stmt compute = lower(stmt, "compute",  false, true);
  irp.print(compute);
  cout << endl;

  cout << "----------------SPATIAL LLIR-----------------" << endl;
  set_Spatial_codegen_enabled(true);
  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);
  ir::Stmt computes = lower(stmt, "Plus3",  false, true);
  irp.print(computes);

  cout << "----------------SPATIAL CODEGEN-----------------" << endl;
  codegen->compile(computes, false);
  set_Spatial_codegen_enabled(false);
}

TEST(spatial, sparse_csf_3D_plus2) {
  set_Spatial_codegen_enabled(true);

  Tensor<int> A("A", {16, 16, 16}, {sparse, sparse, sparse}, taco::MemoryLocation::SpatialSparseDRAM);
  Tensor<int> B("B", {16, 16, 16}, {sparse, sparse, sparse}, taco::MemoryLocation::SpatialSparseDRAM);
  Tensor<int> C("C", {16, 16, 16}, {sparse, sparse, sparse}, taco::MemoryLocation::SpatialSparseDRAM);

  for (int i = 0; i < 16; i++) {
    B.insert({i, i, i}, (int) i);
    C.insert({i, i, i}, (int) i);
  }

  IndexVar i("i"), j("j"), k("k");
  A(i, j, k) = B(i, j, k) + C(i, j, k);

  IndexStmt stmt = A.getAssignment().concretize();

  cout << "----------------Post-Schedule Stmt-----------------" << endl;
  cout << stmt << endl;

  ir::IRPrinter irp = ir::IRPrinter(cout);

  cout << "----------------CPU LLIR-----------------" << endl;
  ir::Stmt compute = lower(stmt, "compute",  false, true);
  irp.print(compute);
  cout << endl;

  cout << "----------------SPATIAL LLIR-----------------" << endl;
  set_Spatial_codegen_enabled(true);
  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);
  ir::Stmt computes = lower(stmt, "Plus2CSF",  false, true);
  irp.print(computes);

  cout << "----------------SPATIAL CODEGEN-----------------" << endl;
  set_tensor_files(true);
  codegen->compile(computes, false);
  set_Spatial_codegen_enabled(false);
  set_tensor_files(false);
}

TEST(spatial, sparse_dss_3D_plus2) {
  set_Spatial_codegen_enabled(true);

  Tensor<int> A("A", {16, 16, 16}, {dense, sparse, sparse}, taco::MemoryLocation::SpatialSparseDRAM);
  Tensor<int> B("B", {16, 16, 16}, {dense, sparse, sparse}, taco::MemoryLocation::SpatialSparseDRAMFalse);
  Tensor<int> C("C", {16, 16, 16}, {dense, sparse, sparse}, taco::MemoryLocation::SpatialSparseDRAMFalse);

  for (int i = 0; i < 16; i++) {
    B.insert({i, i, i}, (int) i);
    C.insert({i, i, i}, (int) i);
  }

  IndexVar i("i"), j("j"), k("k");
  A(i, j, k) = B(i, j, k) + C(i, j, k);

  IndexStmt stmt = A.getAssignment().concretize();

  cout << "----------------Post-Schedule Stmt-----------------" << endl;
  cout << stmt << endl;

  ir::IRPrinter irp = ir::IRPrinter(cout);

  cout << "----------------CPU LLIR-----------------" << endl;
  ir::Stmt compute = lower(stmt, "compute",  false, true);
  irp.print(compute);
  cout << endl;

  cout << "----------------SPATIAL LLIR-----------------" << endl;
  set_Spatial_codegen_enabled(true);
  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);
  ir::Stmt computes = lower(stmt, "Plus2CSF_DSS",  false, true);
  irp.print(computes);

  cout << "----------------SPATIAL CODEGEN-----------------" << endl;
  set_tensor_files(true);
  codegen->compile(computes, false);
  set_Spatial_codegen_enabled(false);
  set_tensor_files(false);
}

TEST(spatial, csr_residual) {
  set_Spatial_codegen_enabled(false);

  Tensor<int> y("y", {16}, {Dense}, taco::MemoryLocation::SpatialFIFO);
  Tensor<int> b("b", {16}, {Dense}, taco::MemoryLocation::SpatialFIFO);
  Tensor<int> A("A", {16, 16}, CSR, taco::MemoryLocation::SpatialFIFO);
  Tensor<int> x("x", {16}, {Dense}, taco::MemoryLocation::SpatialSparseSRAM);

  for (int i = 0; i < 16; i++) {
    if (i % 4 == 0) {
      b.insert({i}, (int) i);
      x.insert({i}, (int) i);
    }
    A.insert({i, i}, (int) i);

  }

  IndexVar i("i"), j("j");
  y(i) = b(i) - A(i, j)*x(j);

  IndexStmt stmt = y.getAssignment().concretize();
  stmt = stmt.parallelize(j, ParallelUnit::Spatial, OutputRaceStrategy::SpatialReduction);
  stmt = scalarPromote(stmt);
  cout << "----------------Post-Schedule Stmt-----------------" << endl;
  cout << stmt << endl;

  ir::IRPrinter irp = ir::IRPrinter(cout);

  cout << "----------------CPU LLIR-----------------" << endl;
  ir::Stmt compute = lower(stmt, "compute",  false, true);
  irp.print(compute);
  cout << endl;

  cout << "----------------SPATIAL LLIR-----------------" << endl;
  set_Spatial_codegen_enabled(true);
  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);
  ir::Stmt computes = lower(stmt, "Residual",  false, true);
  irp.print(computes);

  cout << "----------------SPATIAL CODEGEN-----------------" << endl;
  codegen->compile(computes, false);
  set_Spatial_codegen_enabled(false);
}

TEST(spatial, csr_residual_op) {
  set_Spatial_codegen_enabled(false);

  Tensor<int> y("y", {16}, {Dense}, taco::MemoryLocation::SpatialSRAM);
  Tensor<int> b("b", {16}, {Dense}, taco::MemoryLocation::SpatialSparseParSRAMSwizzle);
  Tensor<int> A("A", {16, 16}, CSR, taco::MemoryLocation::SpatialFIFORetimed);
  Tensor<int> x("x", {16}, {Dense}, taco::MemoryLocation::SpatialSparseParSRAMSwizzle);

  for (int i = 0; i < 16; i++) {
    if (i % 4 == 0) {
      b.insert({i}, (int) i);
      x.insert({i}, (int) i);
    }
    A.insert({i, i}, (int) i);

  }

  IndexVar i("i"), j("j");
  y(i) = b(i) - A(i, j)*x(j);

  IndexStmt stmt = y.getAssignment().concretize();
  stmt = stmt.parallelize(j, ParallelUnit::Spatial, OutputRaceStrategy::SpatialReduction, 16);
  stmt = scalarPromote(stmt);
  stmt = stmt.environment("bp", 2);
  cout << stmt << endl;
  stmt = stmt.communicate(A(i, j), j);

  cout << "----------------Post-Schedule Stmt-----------------" << endl;
  cout << stmt << endl;

  ir::IRPrinter irp = ir::IRPrinter(cout);

  cout << "----------------CPU LLIR-----------------" << endl;
  ir::Stmt compute = lower(stmt, "compute",  false, true);
  irp.print(compute);
  cout << endl;

  cout << "----------------SPATIAL LLIR-----------------" << endl;
  set_Spatial_codegen_enabled(true);
  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);
  ir::Stmt computes = lower(stmt, "Residual_OP",  false, true);
  irp.print(computes);

  cout << "----------------SPATIAL CODEGEN-----------------" << endl;
  codegen->compile(computes, false);
  set_Spatial_codegen_enabled(false);
}

TEST(spatial, sparse_csr_SDDMM) {
  set_Spatial_codegen_enabled(false);
  int N = 16;
  Format   rm({Dense,Dense});
  Format   cm({Dense,Dense}, {1,0});
  Tensor<int> A("A", {N, N}, CSR, taco::MemoryLocation::SpatialFIFO);
  Tensor<int> B("B", {N, N}, CSR, taco::MemoryLocation::SpatialFIFO);
  Tensor<int> C("C", {N, N}, rm, taco::MemoryLocation::SpatialSRAM);
  Tensor<int> D("D", {N, N}, cm, taco::MemoryLocation::SpatialSRAM);

  for (int i = 0; i < N; i++) {
    B.insert({i, i}, (int) i);
    C.insert({i, i}, (int) i);
    D.insert({i, i}, (int) i);
  }

  IndexVar i("i"), j("j"), k("k");
  A(i, j) = B(i, j) * C(i, k)*D(k, j);

  IndexStmt stmt = A.getAssignment().concretize();
  stmt = stmt.parallelize(k, ParallelUnit::Spatial, OutputRaceStrategy::SpatialReduction);
  stmt = scalarPromote(stmt);
  cout << "----------------Post-Schedule Stmt-----------------" << endl;


  TensorVar tjA("tjA", Type(Int32), {}, MemoryLocation::SpatialReg);
  IndexStmt stmt1 = forall(i, forall(j, where(A(i,j) = tjA(), forall(k, tjA() += B(i,j) * C(i,k) * D(k,j), ParallelUnit::Spatial, OutputRaceStrategy::SpatialReduction))));
  stmt1 = stmt1.communicate(C(i,k), k);
  stmt1 = stmt1.communicate(D(k,j), k);

  cout << stmt1 << endl;

  ir::IRPrinter irp = ir::IRPrinter(cout);

  cout << "----------------CPU LLIR-----------------" << endl;
  ir::Stmt compute = lower(stmt, "compute",  false, true);
  irp.print(compute);
  cout << endl;

  cout << "----------------SPATIAL LLIR-----------------" << endl;
  set_Spatial_codegen_enabled(true);
  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);
  ir::Stmt computes = lower(stmt1, "SDDMM",  false, true);
  irp.print(computes);

  cout << "----------------SPATIAL CODEGEN-----------------" << endl;
  codegen->compile(computes, false);
  set_Spatial_codegen_enabled(false);
}

TEST(spatial, sparse_csr_SDDMM_ws) {
  set_Spatial_codegen_enabled(false);
  int N = 16;
  Format   rm({Dense,Dense});
  Format   cm({Dense,Dense}, {1,0});
  Tensor<int> A("A", {N, N}, CSR, taco::MemoryLocation::SpatialFIFO);
  Tensor<int> B("B", {N, N}, CSR, taco::MemoryLocation::SpatialFIFO);
  Tensor<int> C("C", {N, N}, rm, taco::MemoryLocation::SpatialDRAM);
  Tensor<int> D("D", {N, N}, cm, taco::MemoryLocation::SpatialDRAM);

  for (int i = 0; i < N; i++) {
    B.insert({i, i}, (int) i);
    C.insert({i, i}, (int) i);
    D.insert({i, i}, (int) i);
  }

  IndexVar i("i"), j("j"), k("k");
  A(i, j) = B(i, j) * C(i, k)*D(k, j);

  IndexStmt stmt = A.getAssignment().concretize();
  stmt = stmt.parallelize(k, ParallelUnit::Spatial, OutputRaceStrategy::SpatialReduction);
  stmt = scalarPromote(stmt);
  cout << "----------------Post-Schedule Stmt-----------------" << endl;


  TensorVar tjA("tjA", Type(Int32), {}, MemoryLocation::SpatialReg);
  TensorVar C_ws("C_ws", Type(Int32, {N}), taco::Dense , MemoryLocation::SpatialSRAM);
  TensorVar D_ws("D_ws", Type(Int32, {N}), taco::Dense, MemoryLocation::SpatialSRAM);
  IndexStmt stmt1 = forall(i, forall(j, where(A(i,j) = tjA(), forall(k, tjA() += B(i,j) * C(i,k) * D(k,j), ParallelUnit::Spatial, OutputRaceStrategy::SpatialReduction))));

  stmt1 = forall(i, forall(j, where(A(i,j) = tjA(), where(where(forall(k, tjA() += B(i,j) * C_ws(k) * D_ws(k),
                  ParallelUnit::Spatial, OutputRaceStrategy::SpatialReduction), forall(k, C_ws(k) = C(i,k))), forall(k, D_ws(k) = D(k,j))))
                                                                 ));


  cout << stmt1 << endl;

  ir::IRPrinter irp = ir::IRPrinter(cout);

  cout << "----------------CPU LLIR-----------------" << endl;
  ir::Stmt compute = lower(stmt, "compute",  false, true);
  irp.print(compute);
  cout << endl;

  cout << "----------------SPATIAL LLIR-----------------" << endl;
  set_Spatial_codegen_enabled(true);
  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);
  ir::Stmt computes = lower(stmt1, "SDDMM",  false, true);
  irp.print(computes);

  cout << "----------------SPATIAL CODEGEN-----------------" << endl;
  codegen->compile(computes, false);
  set_Spatial_codegen_enabled(false);
}

TEST(spatial, sparse_csr_SDDMM_ws_op) {
  set_Spatial_codegen_enabled(false);
  int N = 16;
  Format   rm({Dense,Dense});
  Format   cm({Dense,Dense}, {1,0});
  Tensor<int> A("A", {N, N}, CSR, taco::MemoryLocation::SpatialFIFO);
  Tensor<int> B("B", {N, N}, CSR, taco::MemoryLocation::SpatialFIFO);
  Tensor<int> C("C", {N, N}, rm, taco::MemoryLocation::SpatialDRAM);
  Tensor<int> D("D", {N, N}, cm, taco::MemoryLocation::SpatialDRAM);

  for (int i = 0; i < N; i++) {
    B.insert({i, i}, (int) i);
    C.insert({i, i}, (int) i);
    D.insert({i, i}, (int) i);
  }

  IndexVar i("i"), j("j"), k("k");
  A(i, j) = B(i, j) * C(i, k)*D(k, j);

  IndexStmt stmt = A.getAssignment().concretize();
  stmt = stmt.parallelize(k, ParallelUnit::Spatial, OutputRaceStrategy::SpatialReduction);
  stmt = scalarPromote(stmt);
  cout << "----------------Post-Schedule Stmt-----------------" << endl;


  TensorVar tjA("tjA", Type(Int32), {}, MemoryLocation::SpatialReg);
  TensorVar C_ws("C_ws", Type(Int32, {N}), taco::Dense , MemoryLocation::SpatialSRAM);
  TensorVar D_ws("D_ws", Type(Int32, {N}), taco::Dense, MemoryLocation::SpatialSRAM);
  IndexStmt stmt1 = forall(i, forall(j, where(A(i,j) = tjA(), forall(k, tjA() += B(i,j) * C(i,k) * D(k,j), ParallelUnit::Spatial, OutputRaceStrategy::SpatialReduction))));

  stmt1 = forall(i, forall(j, where(A(i,j) = tjA(), where(where(forall(k, tjA() += B(i,j) * C_ws(k) * D_ws(k),
                                                                       ParallelUnit::Spatial, OutputRaceStrategy::SpatialReduction), forall(k, C_ws(k) = C(i,k))), forall(k, D_ws(k) = D(k,j))))
  ));
  stmt1 = stmt1.environment("bp", 2);
  stmt1 = stmt1.communicate(A(i,j), j);

  cout << stmt1 << endl;

  ir::IRPrinter irp = ir::IRPrinter(cout);

  cout << "----------------CPU LLIR-----------------" << endl;
  ir::Stmt compute = lower(stmt, "compute",  false, true);
  irp.print(compute);
  cout << endl;

  cout << "----------------SPATIAL LLIR-----------------" << endl;
  set_Spatial_codegen_enabled(true);
  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);
  ir::Stmt computes = lower(stmt1, "SDDMM_OP",  false, true);
  irp.print(computes);

  cout << "----------------SPATIAL CODEGEN-----------------" << endl;
  codegen->compile(computes, false);
  set_Spatial_codegen_enabled(false);
}

TEST(spatial, csr_mattransmul) {
  set_Spatial_codegen_enabled(false);
  int N = 16;
  Tensor<int> y("y", {N}, dense, taco::MemoryLocation::SpatialFIFO);
  Tensor<int> A("A", {N, N}, CSC, taco::MemoryLocation::SpatialFIFO);
  Tensor<int> alpha("alpha", {}, {}, taco::MemoryLocation::SpatialArgIn);
  Tensor<int> x("x", {N}, dense, taco::MemoryLocation::SpatialSparseSRAM);
  Tensor<int> beta("beta", {}, {}, taco::MemoryLocation::SpatialArgIn);
  Tensor<int> z("z", {N}, dense, taco::MemoryLocation::SpatialFIFO);

  for (int i = 0; i < N; i++) {
    if (i % 4 == 0) {
      z.insert({i}, (int) i);
      x.insert({i}, (int) i);
    }
    A.insert({i, i}, (int) i);
  }
  alpha = 1;
  beta = 2;

  IndexVar i("i"), j("j");
  y(i) = alpha()*A(j, i)*x(j) + beta()*z(i);

  IndexStmt stmt = y.getAssignment().concretize();
  stmt = stmt.parallelize(j, ParallelUnit::Spatial, OutputRaceStrategy::SpatialReduction);
  stmt = scalarPromote(stmt);
  cout << "----------------Post-Schedule Stmt-----------------" << endl;
  cout << stmt << endl;

  ir::IRPrinter irp = ir::IRPrinter(cout);

  cout << "----------------CPU LLIR-----------------" << endl;
  ir::Stmt compute = lower(stmt, "compute",  false, true);
  irp.print(compute);
  cout << endl;

  cout << "----------------SPATIAL LLIR-----------------" << endl;
  set_Spatial_codegen_enabled(true);
  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);
  ir::Stmt computes = lower(stmt, "MatTransMul",  false, true);
  irp.print(computes);

  cout << "----------------SPATIAL CODEGEN-----------------" << endl;
  codegen->compile(computes, false);
  set_Spatial_codegen_enabled(false);
}

TEST(spatial, csr_mattransmul_op) {
  set_Spatial_codegen_enabled(false);

  int N = 32;
  Tensor<int> y("y", {N}, dense, taco::MemoryLocation::SpatialSRAM);
  Tensor<int> A("A", {N, N}, CSC, taco::MemoryLocation::SpatialFIFORetimed);
  Tensor<int> alpha("alpha", {}, {}, taco::MemoryLocation::SpatialArgIn);
  Tensor<int> x("x", {N}, dense, taco::MemoryLocation::SpatialSparseParSRAMSwizzle);
  Tensor<int> beta("beta", {}, {}, taco::MemoryLocation::SpatialArgIn);
  Tensor<int> z("z", {N}, dense, taco::MemoryLocation::SpatialSparseParSRAMSwizzle);

  for (int i = 0; i < N; i++) {
    if (i % 4 == 0) {
      z.insert({i}, (int) i);
      x.insert({i}, (int) i);
    }
    A.insert({i, i}, (int) i);
  }
  alpha = 1;
  beta = 2;

  IndexVar i("i"), j("j");
  y(i) = alpha()*A(j, i)*x(j) + beta()*z(i);

  IndexStmt stmt = y.getAssignment().concretize();
  stmt = stmt.parallelize(j, ParallelUnit::Spatial, OutputRaceStrategy::SpatialReduction, 16);
  stmt = scalarPromote(stmt);
  stmt = stmt.environment("bp", 2);
  cout << stmt << endl;
  stmt = stmt.communicate(A(i, j), j);
  cout << "----------------Post-Schedule Stmt-----------------" << endl;
  cout << stmt << endl;

  ir::IRPrinter irp = ir::IRPrinter(cout);

  cout << "----------------CPU LLIR-----------------" << endl;
  ir::Stmt compute = lower(stmt, "compute",  false, true);
  irp.print(compute);
  cout << endl;

  cout << "----------------SPATIAL LLIR-----------------" << endl;
  set_Spatial_codegen_enabled(true);
  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);
  ir::Stmt computes = lower(stmt, "MatTransMul_OP",  false, true);
  irp.print(computes);

  cout << "----------------SPATIAL CODEGEN-----------------" << endl;
  codegen->compile(computes, false);
  set_Spatial_codegen_enabled(false);
}

TEST(spatial, sparse_csf_3D_TTV) {
  set_Spatial_codegen_enabled(false);

  Tensor<int> A("A", {16, 16}, {sparse, sparse}, taco::MemoryLocation::SpatialFIFO);
  Tensor<int> B("B", {16, 16, 16}, {sparse, sparse, sparse}, taco::MemoryLocation::SpatialFIFORetimed);
  Tensor<int> c("c", {16}, {dense}, taco::MemoryLocation::SpatialSparseSRAM);

  for (int i = 0; i < 16; i++) {
    B.insert({i, i, i}, (int) i);
    c.insert({i}, (int) i);
  }

  IndexVar i("i"), j("j"), k("k");
  A(i, j) = B(i, j, k) * c(k);

  IndexStmt stmt = A.getAssignment().concretize();
  stmt = stmt.parallelize(k, ParallelUnit::Spatial, OutputRaceStrategy::SpatialReduction, 16);
  stmt = scalarPromote(stmt);

  TensorVar A_ws("A_ws", Type(Float64, {16}), taco::sparse, MemoryLocation::SpatialFIFO);
  IndexExpr preExpr = stmt.as<Forall>().getStmt().as<Forall>().getStmt().as<Where>().getConsumer().as<Assignment>().getRhs();
  cout << preExpr << endl;
  stmt = stmt.communicate(A(i,j), j);
  cout << "----------------Post-Schedule Stmt-----------------" << endl;
  cout << stmt << endl;

  ir::IRPrinter irp = ir::IRPrinter(cout);

  cout << "----------------CPU LLIR-----------------" << endl;
  ir::Stmt compute = lower(stmt, "compute",  false, true);
  irp.print(compute);
  cout << endl;

  cout << "----------------SPATIAL LLIR-----------------" << endl;
  set_Spatial_codegen_enabled(true);
  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);
  ir::Stmt computes = lower(stmt, "TTV",  false, true);
  irp.print(computes);

  cout << "----------------SPATIAL CODEGEN-----------------" << endl;
  set_tensor_files(true);
  codegen->compile(computes, false);
  set_Spatial_codegen_enabled(false);
  set_tensor_files(false);
}

TEST(spatial, sparse_csf_3D_TTV_op) {
  set_Spatial_codegen_enabled(false);

  Tensor<int> A("A", {16, 16}, {sparse, sparse}, taco::MemoryLocation::SpatialFIFO);
  Tensor<int> B("B", {16, 16, 16}, {sparse, sparse, sparse}, taco::MemoryLocation::SpatialFIFORetimed);
  Tensor<int> c("c", {16}, {dense}, taco::MemoryLocation::SpatialSparseParSRAMSwizzle);

  for (int i = 0; i < 16; i++) {
    B.insert({i, i, i}, (int) i);
    c.insert({i}, (int) i);
  }

  IndexVar i("i"), j("j"), k("k");
  A(i, j) = B(i, j, k) * c(k);

  IndexStmt stmt = A.getAssignment().concretize();
  stmt = stmt.parallelize(k, ParallelUnit::Spatial, OutputRaceStrategy::SpatialReduction, 16);
  stmt = scalarPromote(stmt);

  TensorVar A_ws("A_ws", Type(Float64, {16}), taco::sparse, MemoryLocation::SpatialFIFO);
  IndexExpr preExpr = stmt.as<Forall>().getStmt().as<Forall>().getStmt().as<Where>().getConsumer().as<Assignment>().getRhs();
  cout << preExpr << endl;
  stmt = stmt.communicate(A(i,j), j);
  stmt = stmt.environment("bp", 2);
  cout << "----------------Post-Schedule Stmt-----------------" << endl;
  cout << stmt << endl;

  ir::IRPrinter irp = ir::IRPrinter(cout);

  cout << "----------------CPU LLIR-----------------" << endl;
  ir::Stmt compute = lower(stmt, "compute",  false, true);
  irp.print(compute);
  cout << endl;

  cout << "----------------SPATIAL LLIR-----------------" << endl;
  set_Spatial_codegen_enabled(true);
  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);
  ir::Stmt computes = lower(stmt, "TTV_OP",  false, true);
  irp.print(computes);

  cout << "----------------SPATIAL CODEGEN-----------------" << endl;
  set_tensor_files(true);
  codegen->compile(computes, false);
  set_Spatial_codegen_enabled(false);
  set_tensor_files(false);
}

TEST(spatial, sparse_csf_3D_TTM) {
  set_Spatial_codegen_enabled(false);

  Tensor<int> A("A", {16, 16, 16}, {sparse, sparse, sparse}, taco::MemoryLocation::SpatialFIFO);
  Tensor<int> B("B", {16, 16, 16}, {sparse, sparse, sparse}, taco::MemoryLocation::SpatialFIFORetimed);
  Tensor<int> C("C", {16, 16}, {dense, dense}, taco::MemoryLocation::SpatialSparseDRAM);

  for (int i = 0; i < 16; i++) {
    B.insert({i, i, i}, (int) i);
    C.insert({i, i}, (int) i);
  }

  IndexVar i("i"), j("j"), k("k"), l("l");
  A(i, j, k) = B(i, j, l) * C(k, l);

  IndexStmt stmt = A.getAssignment().concretize();
  stmt = stmt.parallelize(l, ParallelUnit::Spatial, OutputRaceStrategy::SpatialReduction, 16);
  stmt = scalarPromote(stmt);
  stmt = stmt.communicate(A(i,j,k), j);
//
//  TensorVar tlA("tlA", Type(Int32), {}, MemoryLocation::SpatialReg);
//  TensorVar A_ws("A_ws", Type(Int32, {16}), taco::dense, MemoryLocation::SpatialFIFO);
//  IndexStmt stmt1 = forall(i, forall(j, where(forall(k, A(i,j,k) = A_ws(k)) , forall(k, where(A_ws(k) = tlA, forall(l, tlA += B(i,j,l) * C(k,l),
//                                                                     taco::ParallelUnit::Spatial, taco::OutputRaceStrategy::SpatialReduction, 0, 16))))));
  cout << "----------------Post-Schedule Stmt-----------------" << endl;
  cout << stmt << endl;

  ir::IRPrinter irp = ir::IRPrinter(cout);

  cout << "----------------SPATIAL LLIR-----------------" << endl;
  set_Spatial_codegen_enabled(true);
  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);
  ir::Stmt computes = lower(stmt, "TTM",  false, true);
  irp.print(computes);

  cout << "----------------SPATIAL CODEGEN-----------------" << endl;
  set_tensor_files(true);
  codegen->compile(computes, false);
  set_Spatial_codegen_enabled(false);
  set_tensor_files(false);
}

TEST(spatial, sparse_csf_3D_TTM_op) {
  set_Spatial_codegen_enabled(false);
  int N = 16;
  Tensor<int> A("A", {N, N, N}, {sparse, sparse, sparse}, taco::MemoryLocation::SpatialFIFO);
  Tensor<int> B("B", {N, N, N}, {sparse, sparse, sparse}, taco::MemoryLocation::SpatialFIFORetimed);
  Tensor<int> C("C", {N, N}, {dense, dense}, taco::MemoryLocation::SpatialSparseDRAM);

  for (int i = 0; i < N; i++) {
    B.insert({i, i, i}, (int) i);
    C.insert({i, i}, (int) i);
  }

  IndexVar i("i"), j("j"), k("k"), l("l");
  A(i, j, k) = B(i, j, l) * C(k, l);

  IndexStmt stmt = A.getAssignment().concretize();
  stmt = stmt.parallelize(l, ParallelUnit::Spatial, OutputRaceStrategy::SpatialReduction, 16);
  stmt = scalarPromote(stmt);
  stmt = stmt.communicate(A(i,j,k), j);
  stmt = stmt.environment("bp", 2);

  cout << "----------------Post-Schedule Stmt-----------------" << endl;
  cout << stmt << endl;

  ir::IRPrinter irp = ir::IRPrinter(cout);

  cout << "----------------SPATIAL LLIR-----------------" << endl;
  set_Spatial_codegen_enabled(true);
  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);
  ir::Stmt computes = lower(stmt, "TTM_OP",  false, true);
  irp.print(computes);

  cout << "----------------SPATIAL CODEGEN-----------------" << endl;
  set_tensor_files(true);
  codegen->compile(computes, false);
  set_Spatial_codegen_enabled(false);
  set_tensor_files(false);
}

TEST(spatial, sparse_csf_3D_MTTKRP) {
  set_Spatial_codegen_enabled(false);

  int N = 16;
  Tensor<int> A("A", {N, N}, {sparse, dense}, taco::MemoryLocation::SpatialFIFO);
  Tensor<int> B("B", {N, N, N}, {sparse, sparse, sparse}, taco::MemoryLocation::SpatialFIFORetimed);
  Tensor<int> C("C", {N, N}, {dense, dense}, taco::MemoryLocation::SpatialSparseDRAM);
  Tensor<int> D("D", {N, N}, {dense, dense}, taco::MemoryLocation::SpatialSparseDRAM);

  for (int i = 0; i < 16; i++) {
    B.insert({i, i, i}, (int) i);
    C.insert({i, i}, (int) i);
  }

  IndexVar i("i"), j("j"), k("k"), l("l");
  A(i, j) = B(i, k, l) * C(k, j) * D(l, j);

  TensorVar tkA("tkA", Type(Int32), {}, MemoryLocation::SpatialReg);
  TensorVar Ct("Ct", Type(Int32, {N}), taco::Dense , MemoryLocation::SpatialSRAM);
  TensorVar Dt("Dt", Type(Int32, {N}), taco::Dense, MemoryLocation::SpatialSRAM);

  IndexStmt stmt = A.getAssignment().concretize();
//  stmt = stmt.reorder({i, k, l, j});
//  IndexStmt stmt1 = forall(i, where(forall(j, A(i,j) = tkA) ,
//                                    forall(k, forall(l, forall(j, tkA += B(i,k,l) * C(k,j) * D(l,j), ParallelUnit::Spatial,
//                                                               OutputRaceStrategy::SpatialReduction)))));
//  stmt1 = stmt1.parallelize(l, ParallelUnit::Spatial, OutputRaceStrategy::SpatialReduction, 16);
//  stmt1 = stmt1.parallelize(k, ParallelUnit::Spatial, OutputRaceStrategy::SpatialReduction, 16);
//  //stmt1 = stmt1.parallelize(j, ParallelUnit::Spatial, OutputRaceStrategy::SpatialReduction, 16);
//  stmt1 = stmt1.communicate(A(i,j), i);
//  stmt1 = stmt1.communicate(C(k,j), )

  stmt = stmt.parallelize(l, ParallelUnit::Spatial, OutputRaceStrategy::SpatialReduction);
  stmt = stmt.parallelize(k, ParallelUnit::Spatial, OutputRaceStrategy::SpatialReduction);
  stmt = stmt.parallelize(j, ParallelUnit::Spatial, OutputRaceStrategy::SpatialReduction);
  stmt = scalarPromote(stmt);
  stmt = stmt.communicate(A(i,j), i);

  cout << "----------------Post-Schedule Stmt-----------------" << endl;
  cout << stmt << endl;

  ir::IRPrinter irp = ir::IRPrinter(cout);


  cout << "----------------SPATIAL LLIR-----------------" << endl;
  set_Spatial_codegen_enabled(true);
  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);
  ir::Stmt computes = lower(stmt, "MTTKRP",  false, true);
  irp.print(computes);

  cout << "----------------SPATIAL CODEGEN-----------------" << endl;
  set_tensor_files(true);
  codegen->compile(computes, false);
  set_Spatial_codegen_enabled(false);
  set_tensor_files(false);
}

TEST(spatial, sparse_csf_3D_MTTKRP_op) {
  set_Spatial_codegen_enabled(false);

  int N = 16;
  Tensor<int> A("A", {N, N}, {sparse, dense}, taco::MemoryLocation::SpatialFIFO);
  Tensor<int> B("B", {N, N, N}, {sparse, sparse, sparse}, taco::MemoryLocation::SpatialFIFORetimed);
  Tensor<int> C("C", {N, N}, {dense, dense}, taco::MemoryLocation::SpatialSparseDRAM);
  Tensor<int> D("D", {N, N}, {dense, dense}, taco::MemoryLocation::SpatialSparseDRAM);

  for (int i = 0; i < 16; i++) {
    B.insert({i, i, i}, (int) i);
    C.insert({i, i}, (int) i);
  }

  IndexVar i("i"), j("j"), k("k"), l("l");
  A(i, j) = B(i, k, l) * C(k, j) * D(l, j);

  IndexStmt stmt = A.getAssignment().concretize();

  stmt = stmt.parallelize(l, ParallelUnit::Spatial, OutputRaceStrategy::SpatialReduction);
  stmt = stmt.parallelize(k, ParallelUnit::Spatial, OutputRaceStrategy::SpatialReduction);
  stmt = stmt.parallelize(j, ParallelUnit::Spatial, OutputRaceStrategy::SpatialReduction);
  stmt = scalarPromote(stmt);
  stmt = stmt.communicate(A(i,j), j);
  stmt = stmt.environment("bp", 2);

  cout << "----------------Post-Schedule Stmt-----------------" << endl;
  cout << stmt << endl;

  ir::IRPrinter irp = ir::IRPrinter(cout);


  cout << "----------------SPATIAL LLIR-----------------" << endl;
  set_Spatial_codegen_enabled(true);
  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);
  ir::Stmt computes = lower(stmt, "MTTKRP_OP",  false, true);
  irp.print(computes);

  cout << "----------------SPATIAL CODEGEN-----------------" << endl;
  set_tensor_files(true);
  codegen->compile(computes, false);
  set_Spatial_codegen_enabled(false);
  set_tensor_files(false);
}

TEST(spatial, sparse_csf_3D_innerprod) {
  set_Spatial_codegen_enabled(false);

  Tensor<int> a("a", {}, {}, MemoryLocation::SpatialArgOut);
  Tensor<int> B("B", {16, 16, 16}, {sparse, sparse, sparse}, taco::MemoryLocation::SpatialSparseDRAM);
  Tensor<int> C("C", {16, 16, 16}, {sparse, sparse, sparse}, taco::MemoryLocation::SpatialSparseDRAM);

  for (int i = 0; i < 16; i++) {
    B.insert({i, i, i}, (int) i);
    C.insert({i, i, i}, (int) i);
  }

  IndexVar i("i"), j("j"), k("k"), l("l");
  a() = B(i, j, k) * C(i, j, k);

  IndexStmt stmt = a.getAssignment().concretize();
  stmt = stmt.parallelize(i, ParallelUnit::Spatial,OutputRaceStrategy::SpatialReduction);
  stmt = stmt.parallelize(j, ParallelUnit::Spatial,OutputRaceStrategy::SpatialReduction);
  stmt = stmt.parallelize(k, ParallelUnit::Spatial,OutputRaceStrategy::SpatialReduction);
  stmt = stmt.environment("sp", 1);
  cout << "----------------Post-Schedule Stmt-----------------" << endl;
  cout << stmt << endl;

  ir::IRPrinter irp = ir::IRPrinter(cout);


  cout << "----------------SPATIAL LLIR-----------------" << endl;
  set_Spatial_codegen_enabled(true);
  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);
  ir::Stmt computes = lower(stmt, "InnerProd",  false, true);
  irp.print(computes);

  cout << "----------------SPATIAL CODEGEN-----------------" << endl;
  set_tensor_files(true);
  codegen->compile(computes, false);
  set_Spatial_codegen_enabled(false);
  set_tensor_files(false);
}

TEST(spatial, sparse_dss_3D_innerprod) {
  set_Spatial_codegen_enabled(false);

  Tensor<int> a("a", {}, {}, MemoryLocation::SpatialArgOut);
  Tensor<int> B("B", {16, 16, 16}, {dense, sparse, sparse}, taco::MemoryLocation::SpatialSparseDRAM);
  Tensor<int> C("C", {16, 16, 16}, {dense, sparse, sparse}, taco::MemoryLocation::SpatialSparseDRAM);

  for (int i = 0; i < 16; i++) {
    B.insert({i, i, i}, (int) i);
    C.insert({i, i, i}, (int) i);
  }

  IndexVar i("i"), j("j"), k("k"), l("l");
  a() = B(i, j, k) * C(i, j, k);

  IndexStmt stmt = a.getAssignment().concretize();
  stmt = stmt.parallelize(i, ParallelUnit::Spatial,OutputRaceStrategy::SpatialReduction);
  stmt = stmt.parallelize(j, ParallelUnit::Spatial,OutputRaceStrategy::SpatialReduction);
  stmt = stmt.parallelize(k, ParallelUnit::Spatial,OutputRaceStrategy::SpatialReduction);
  stmt = stmt.environment("sp", 1);
  cout << "----------------Post-Schedule Stmt-----------------" << endl;
  cout << stmt << endl;

  ir::IRPrinter irp = ir::IRPrinter(cout);


  cout << "----------------SPATIAL LLIR-----------------" << endl;
  set_Spatial_codegen_enabled(true);
  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);
  ir::Stmt computes = lower(stmt, "InnerProdDSS",  false, true);
  irp.print(computes);

  cout << "----------------SPATIAL CODEGEN-----------------" << endl;
  set_tensor_files(true);
  codegen->compile(computes, false);
  set_Spatial_codegen_enabled(false);
  set_tensor_files(false);
}

TEST(spatial, tian_cpu) {
  std::cout << "Format: ./sparse_csr_SDDMM mat0 " << std::endl;
  int repeat = 1;
  int rowsB, colsB;

  // cout << "Parsing " << argv[1] << endl;
  string ss;
  ss = "/Users/oliviahsu/Files/research/sparse_spatial_compilation/taco-to-spatial/data/mats/ckt11752_dc_1.mtx";
  // readMatrixSize(ss, rowsB, colsB);
  // cout << "#rows = " << rowsB << ", #cols = " << colsB << endl;
  Tensor<double> B = read(ss, CSR, true);
  int N = B.getDimensions()[0];
  cout << "N = "  << N << endl;

  Format rm({Dense, Dense}); // The last two tensors should be dense?
  Format cm({Dense, Dense}, {1, 0});
  Tensor<double> A("A", {N, N}, CSR);
  Tensor<double> C("C", {N, N}, rm);
  Tensor<double> D("D", {N, N}, cm);

  IndexVar i("i"), j("j"), k("k");
  A(i, j) = B(i, j) * C(i, k) * D(k, j);

  A.compile();
  A.assemble();
  A.compute();

  cout << A;

  // write(std::string("./result.tns"), FileType::tns, y);
}
