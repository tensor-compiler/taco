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

TEST(spatial, tile_vecElemMul) {
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
  stmt = stmt.bound(i, i_b, 16, BoundType::MaxExact)
             .parallelize(i_b, ParallelUnit::Spatial, OutputRaceStrategy::SpatialReduction);

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
  cout << A << endl;
  cout << expected << endl;
  set_Spatial_codegen_enabled(true);
//
  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);
  ir::Stmt compute = lower(stmt, "Compute",  false, true);
  cout << "finish codegen lowering" << endl;
  cout << compute << endl;
//  irp.print(compute);
//  cout << endl;
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

  // [Spatial] TODO: Add in temporary workspace
  IndexStmt stmt = A.getAssignment().concretize();
  stmt = stmt
          .bound(i, i_b, 16, BoundType::MaxExact)
          .bound(j, j_b, 16, BoundType::MaxExact)
          .parallelize(i_b, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces)
          .parallelize(j_b, ParallelUnit::Spatial, OutputRaceStrategy::SpatialReduction);

  stmt = scalarPromote(stmt);

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

TEST(spatial, GEMV) {

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

  // [Spatial] TODO: Add in temporary workspace
  IndexStmt stmt = A.getAssignment().concretize();

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

  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);
  ir::Stmt compute = lower(stmt, "Compute", false, true);
  cout << "----------Finish codegen lowering---------" << endl;
  cout << compute << endl;

  cout << "-----------Spatial Code---------------" << endl;
  codegen->compile(compute, false);
}