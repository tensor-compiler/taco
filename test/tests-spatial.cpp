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
}

TEST(spatial, tileCompute_vecElemMul) {
  // Enable spatial codegen
  //should_use_Spatial_codegen();
  
  Tensor<double> A("A", {16}, {Dense});
  Tensor<double> B("B", {16}, {Dense});
  Tensor<double> C("C", {16}, {Dense});

  for (int i = 0; i < 16; i++) {
      A.insert({i}, (double) i);
      B.insert({i}, (double) i);
  }

  A.pack();
  B.pack();

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
}

TEST(spatial, tile_matElemMul) {

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

}

TEST(spatial, reduction_dotProduct) {

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
  stmt = stmt.bound(i, i_b, 16, BoundType::MaxExact);
             //.parallelize(i_b, ParallelUnit::Spatial,
             //             OutputRaceStrategy::SpatialReduction);

  ir::IRPrinter irp = ir::IRPrinter(cout);
  cout << stmt << endl;

  A.compile(stmt);
  A.assemble();
  A.compute();

  Tensor<int> expected("expected");
  expected() = B(i) * C(i);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(A, expected);

  cout << "----------------Resulting Tensors-----------------" << endl;
  cout << A << endl;
  cout << expected << endl;
  set_Spatial_codegen_enabled(true);

  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);
  ir::Stmt compute = lower(stmt, "Compute",  false, true);

  cout << "----------Finish codegen lowering---------" << endl;
  cout << compute << endl;

  cout << "-----------Spatial Code---------------" << endl;
  codegen->compile(compute, false);

}

TEST(spatial, bound_elemMul) {

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

}

TEST(spatial, reduction_GEMV) {

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
  cout << stmt << endl;

  A.compile(stmt);
  A.assemble();
  A.compute();

  Tensor<int> expected("expected", {16}, {Dense});
  expected(i) = B(i, j) * C(j);
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

TEST(spatial, reduction_GEMM) {

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
          //.parallelize(i_b, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces)
          .parallelize(k_b, ParallelUnit::Spatial, OutputRaceStrategy::SpatialReduction);



  ir::IRPrinter irp = ir::IRPrinter(cout);
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
}

TEST(spatial, outerProduct) {

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
}

TEST(spatial, reduction_higherOrder) {

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
  // Enable spatial codegen
  //should_use_Spatial_codegen();

  Tensor<double> A("A", {16}, {Dense}, MemoryLocation::SpatialDRAM);
  Tensor<double> B("B", {16}, {Dense}, MemoryLocation::SpatialDRAM);
  Tensor<double> C("C", {16}, {Dense}, MemoryLocation::SpatialDRAM);

  for (int i = 0; i < 16; i++) {
    A.insert({i}, (double) i);
    B.insert({i}, (double) i);
  }

  A.pack();
  B.pack();

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

  stmt = stmt.bound(i, i_bounded, 16, BoundType::MaxExact)
          .split(i_bounded, i0, i1, 4)
          .precompute(precomputedExpr, i1, i1, precomputed);
  cout << "----------------Post-Schedule 1 Stmt-----------------" << endl;
  cout << stmt << endl;

  stmt = stmt.precompute(BExpr, i1, i1, B_sram);
  cout << "----------------Post-Schedule 2 Stmt-----------------" << endl;
  cout << stmt << endl;

  stmt = stmt.precompute(CExpr, i1, i1, C_sram);


  cout << "----------------Post-Schedule 3 Stmt-----------------" << endl;
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

  cout << "----------Finish codegen lowering---------" << endl;
  cout << compute << endl;

  codegen->compile(compute, false);
}