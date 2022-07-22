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

TEST(workspaces, tile_vecElemMul_NoTail) {
  
  Tensor<double> A("A", {16}, Format{Dense});
  Tensor<double> B("B", {16}, Format{Dense});
  Tensor<double> C("C", {16}, Format{Dense});

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

  Tensor<double> expected("expected", {16}, Format{Dense});
  expected(i) = B(i) * C(i);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, A);
}

TEST(workspaces, tile_vecElemMul_Tail1) {
  
  Tensor<double> A("A", {16}, Format{Dense});
  Tensor<double> B("B", {16}, Format{Dense});
  Tensor<double> C("C", {16}, Format{Dense});

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
             .split(i_bounded, i0, i1, 5)
             .precompute(precomputedExpr, i1, i1, precomputed);
   
  A.compile(stmt.concretize());
  A.assemble();
  A.compute();

  Tensor<double> expected("expected", {16}, Format{Dense});
  expected(i) = B(i) * C(i);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, A);
}

TEST(workspaces, tile_vecElemMul_Tail2) {
  
  Tensor<double> A("A", {17}, Format{Dense});
  Tensor<double> B("B", {17}, Format{Dense});
  Tensor<double> C("C", {17}, Format{Dense});

  for (int i = 0; i < 17; i++) {
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
  stmt = stmt.bound(i, i_bounded, 17, BoundType::MaxExact)
             .split(i_bounded, i0, i1, 4)
             .precompute(precomputedExpr, i1, i1, precomputed);
   
  A.compile(stmt.concretize());
  A.assemble();
  A.compute();

  Tensor<double> expected("expected", {17}, Format{Dense});
  expected(i) = B(i) * C(i);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, A);

//  ir::IRPrinter irp = ir::IRPrinter(cout);
//    
//  cout << stmt << endl;
//
//  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);
//  ir::Stmt compute = lower(stmt, "compute",  false, true);
//  
//  irp.print(compute);
//  cout << endl;
//  codegen->compile(compute, false);
}

TEST(workspaces, tile_denseMatMul) {
  
  Tensor<double> A("A", {16}, Format{Dense});
  Tensor<double> B("B", {16}, Format{Dense});
  Tensor<double> C("C", {16}, Format{Dense});

  for (int i = 0; i < 16; i++) {
      B.insert({i}, (double) i);
      C.insert({i}, (double) i);
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
             .split(i_bounded, i0, i1, 4);

  stmt = stmt.precompute(precomputedExpr, i1, i1, precomputed);

  A.compile(stmt.concretize());
  A.assemble();
  A.compute();

  Tensor<double> expected("expected", {16}, Format{Dense});
  expected(i) = B(i) * C(i);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, A);

//  ir::IRPrinter irp = ir::IRPrinter(cout);
//    
//  cout << stmt << endl;
//
//  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);
//  ir::Stmt compute = lower(stmt, "compute",  false, true);
//  
//  irp.print(compute);
//  cout << endl;
//  codegen->compile(compute, false);
  
}

TEST(workspaces, precompute2D_add) {
  int N = 16;
  Tensor<double> A("A", {N, N}, Format{Dense, Dense});
  Tensor<double> B("B", {N, N}, Format{Dense, Dense});
  Tensor<double> C("C", {N, N}, Format{Dense, Dense});
  Tensor<double> D("D", {N, N}, Format{Dense, Dense});

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      B.insert({i, j}, (double) i);
      C.insert({i, j}, (double) j);
      D.insert({i, j}, (double) i*j);
    }
  }

  IndexVar i("i"), j("j");
  IndexExpr precomputedExpr = B(i, j) + C(i, j);
  A(i, j) = precomputedExpr + D(i, j);
    
  IndexStmt stmt = A.getAssignment().concretize();
  TensorVar ws("ws", Type(Float64, {(size_t)N, (size_t)N}), Format{Dense, Dense});
  stmt = stmt.precompute(precomputedExpr, {i, j}, {i, j}, ws);

  A.compile(stmt.concretize());
  A.assemble();
  A.compute();

  Tensor<double> expected("expected", {N, N}, Format{Dense, Dense});
  expected(i, j) = B(i, j) + C(i, j) + D(i, j);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, A);

}

TEST(workspaces, precompute4D_add) {
  int N = 16;
  Tensor<double> A("A", {N, N, N, N}, Format{Dense, Dense, Dense, Dense});
  Tensor<double> B("B", {N, N, N, N}, Format{Dense, Dense, Dense, Dense});
  Tensor<double> C("C", {N, N, N, N}, Format{Dense, Dense, Dense, Dense});
  Tensor<double> D("D", {N, N, N, N}, Format{Dense, Dense, Dense, Dense});

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
        for (int l = 0; l < N; l++) {
          B.insert({i, j, k, l}, (double) i + j);
          C.insert({i, j, k, l}, (double) j * k);
          D.insert({i, j, k, l}, (double) k * l);
        }
      }
    }
  }

  IndexVar i("i"), j("j"), k("k"), l("l");
  IndexExpr precomputedExpr = B(i, j, k, l) + C(i, j, k, l);
  A(i, j, k, l) = precomputedExpr + D(i, j, k, l);


  IndexStmt stmt = A.getAssignment().concretize();
  TensorVar ws1("ws1", Type(Float64, {(size_t)N, (size_t)N, (size_t)N, (size_t)N}), 
                Format{Dense, Dense, Dense, Dense});
  TensorVar ws2("ws2", Type(Float64, {(size_t)N, (size_t)N, (size_t)N, (size_t)N}), 
                Format{Dense, Dense, Dense, Dense});
  stmt = stmt.precompute(precomputedExpr, {i, j, k, l}, {i, j, k, l}, ws1)
    .precompute(ws1(i, j, k, l) + D(i, j, k, l), {i, j, k, l}, {i, j, k ,l}, ws2);

  A.compile(stmt.concretize());
  A.assemble();
  A.compute();

  Tensor<double> expected("expected", {N, N, N, N}, Format{Dense, Dense, Dense, Dense});
  expected(i, j, k, l) = B(i, j, k, l) + C(i, j, k, l) + D(i, j, k, l);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, A);
}

TEST(workspaces, precompute4D_multireduce) {
  int N = 16;
  Tensor<double> A("A", {N, N}, Format{Dense, Dense});
  Tensor<double> B("B", {N, N, N, N}, Format{Dense, Dense, Dense, Dense});
  Tensor<double> C("C", {N, N, N}, Format{Dense, Dense, Dense});
  Tensor<double> D("D", {N, N}, Format{Dense, Dense});

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
        for (int l = 0; l < N; l++) {
          B.insert({i, j, k, l}, (double) k*l);
          C.insert({i, j, k}, (double) j * k);
          D.insert({i, j}, (double) i+j);
        }
      }
    }
  }

  IndexVar i("i"), j("j"), k("k"), l("l"), m("m"), n("n");
  IndexExpr precomputedExpr = B(i, j, k, l) * C(k, l, m);
  A(i, j) = precomputedExpr * D(m, n);


  IndexStmt stmt = A.getAssignment().concretize();
  TensorVar ws1("ws1", Type(Float64, {(size_t)N, (size_t)N, (size_t)N}), Format{Dense, Dense, Dense});
  TensorVar ws2("ws2", Type(Float64, {(size_t)N, (size_t)N}), Format{Dense, Dense});
  stmt = stmt.precompute(precomputedExpr, {i, j, m}, {i, j, m}, ws1)
    .precompute(ws1(i, j, m) * D(m, n), {i, j}, {i, j}, ws2);

  A.compile(stmt.concretize());
  A.assemble();
  A.compute();

  Tensor<double> expected("expected", {N, N}, Format{Dense, Dense});
  expected(i, j) = B(i, j, k, l) * C(k, l, m) * D(m, n);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, A);
}

TEST(workspaces, precompute3D_TspV) {
  int N = 16;
  Tensor<double> A("A", {N, N}, Format{Dense, Dense});
  Tensor<double> B("B", {N, N, N, N}, Format{Dense, Dense, Dense, Dense});
  Tensor<double> c("c", {N}, Format{Sparse});

  for (int i = 0; i < N; i++) {
    c.insert({i}, (double) i);
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
        for (int l = 0; l < N; l++) {
          B.insert({i, j, k, l}, (double) i + j);
        }
      }
    }
  }

  IndexVar i("i"), j("j"), k("k"), l("l");
  IndexExpr precomputedExpr = B(i, j, k, l) * c(l);
  A(i, j) = precomputedExpr * c(k);


  IndexStmt stmt = A.getAssignment().concretize();
  TensorVar ws("ws", Type(Float64, {(size_t)N, (size_t)N, (size_t)N}), Format{Dense, Dense, Dense});
  stmt = stmt.precompute(precomputedExpr, {i, j, k}, {i, j, k}, ws);
  stmt = stmt.concretize();

  A.compile(stmt);
  A.assemble();
  A.compute();

  Tensor<double> expected("expected", {N, N}, Format{Dense, Dense});
  expected(i, j) = (B(i, j, k, l) * c(l)) * c(k);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, A);

}

TEST(workspaces, precompute3D_multipleWS) {
  int N = 16;
  Tensor<double> A("A", {N, N}, Format{Dense, Dense});
  Tensor<double> B("B", {N, N, N, N}, Format{Dense, Dense, Dense, Dense});
  Tensor<double> c("c", {N}, Format{Sparse});

  for (int i = 0; i < N; i++) {
    c.insert({i}, (double) i);
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
        for (int l = 0; l < N; l++) {
          B.insert({i, j, k, l}, (double) i + j);
        }
      }
    }
  }

  IndexVar i("i"), j("j"), k("k"), l("l");
  IndexExpr precomputedExpr = B(i, j, k, l) * c(l);
  IndexExpr precomputedExpr2 = precomputedExpr * c(k);
  A(i, j) = precomputedExpr2;


  IndexStmt stmt = A.getAssignment().concretize();
  TensorVar ws("ws", Type(Float64, {(size_t)N, (size_t)N, (size_t)N}), Format{Dense, Dense, Dense});
  TensorVar t("t", Type(Float64, {(size_t) N, (size_t)N}), Format{Dense, Dense});
  stmt = stmt.precompute(precomputedExpr, {i, j, k}, {i, j, k}, ws);

  stmt = stmt.precompute(ws(i, j, k) * c(k), {i, j}, {i, j}, t);
  stmt = stmt.concretize();

  A.compile(stmt);
  A.assemble();
  A.compute();

  Tensor<double> expected("expected", {N, N}, Format{Dense, Dense});
  expected(i, j) = (B(i, j, k, l) * c(l)) * c(k);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, A);

}

TEST(workspaces, precompute3D_renamedIVars_TspV) {
  int N = 16;
  Tensor<double> A("A", {N, N}, Format{Dense, Dense});
  Tensor<double> B("B", {N, N, N, N}, Format{Dense, Dense, Dense, Dense});
  Tensor<double> c("c", {N}, Format{Sparse});

  for (int i = 0; i < N; i++) {
    c.insert({i}, (double) i);
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
        for (int l = 0; l < N; l++) {
          B.insert({i, j, k, l}, (double) i + j);
        }
      }
    }
  }

  IndexVar i("i"), j("j"), k("k"), l("l");
  IndexExpr precomputedExpr = B(i, j, k, l) * c(l);
  A(i, j) = precomputedExpr * c(k);


  IndexStmt stmt = A.getAssignment().concretize();
  TensorVar ws("ws", Type(Float64, {(size_t)N, (size_t)N, (size_t)N}),
               Format{Dense, Dense, Dense});

  IndexVar iw("iw"), jw("jw"), kw("kw");
  stmt = stmt.precompute(precomputedExpr, {i, j, k}, {iw, jw, kw}, ws);
  stmt = stmt.concretize();

  A.compile(stmt);
  A.assemble();
  A.compute();

  Tensor<double> expected("expected", {N, N}, Format{Dense, Dense});
  expected(i, j) = (B(i, j, k, l) * c(l)) * c(k);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, A);

}

TEST(workspaces, tile_dotProduct_1) {
  // Test that precompute algorithm correctly decides the reduction operator of C_new(i1) = C(i) and B_new(i1) = B(i).
  // Current indexStmt is:
  // where(forall(i1, A += precomputed(i1)), forall(i0, where(where(forall(i1, precomputed(i1) += B_new(i1) * C_new(i1))
  // ,forall(i1, C_new(i1) = C(i))), forall(i1, B_new(i1) = B(i)))))

  int N = 1024;
  Tensor<double> A("A");
  Tensor<double> B("B", {N}, Format({Dense}));
  Tensor<double> C("C", {N}, Format({Dense}));

  for (int i = 0; i < N; i++) {
    B.insert({i}, (double) i);
    C.insert({i}, (double) i);
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
  TensorVar B_new("B_new", Type(Float64, {(size_t)N}), taco::dense);
  TensorVar C_new("C_new", Type(Float64, {(size_t)N}), taco::dense);
  TensorVar precomputed("precomputed", Type(Float64, {(size_t)N}), taco::dense);

  stmt = stmt.bound(i, i_bounded, (size_t)N, BoundType::MaxExact)
             .split(i_bounded, i0, i1, 32);
  stmt = stmt.precompute(precomputedExpr, i1, i1, precomputed);
  stmt = stmt.precompute(BExpr, i1, i1, B_new)
    .precompute(CExpr, i1, i1, C_new);
  
  stmt = stmt.concretize();

  A.compile(stmt);
  A.assemble();
  A.compute();

  ir::IRPrinter irp = ir::IRPrinter(cout);
    
  cout << stmt << endl;

  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);
  ir::Stmt compute = lower(stmt, "compute",  false, true);
  
  irp.print(compute);
  cout << endl;
  codegen->compile(compute, false);

  Tensor<double> expected("expected");
  expected() = B(i) * C(i);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, A);
}

TEST(workspaces, tile_dotProduct_2) {
  // Split on the ALL INSTANCES of an indexVar.
  // Test the wsaccel function that can disable the acceleration.

  int N = 1024;
  Tensor<double> A("A");
  Tensor<double> B("B", {N}, Format({Dense}));
  Tensor<double> C("C", {N}, Format({Dense}));

  for (int i = 0; i < N; i++) {
    B.insert({i}, (double) i / N);
    C.insert({i}, (double) i / N);
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
  TensorVar B_new("B_new", Type(Float64, {(size_t)N}), taco::dense);
  TensorVar C_new("C_new", Type(Float64, {(size_t)N}), taco::dense);
  TensorVar precomputed("precomputed", Type(Float64, {(size_t)N}), taco::dense);

  stmt = stmt.precompute(precomputedExpr, i, i, precomputed);
    
  stmt = stmt.precompute(BExpr, i, i, B_new) 
          .precompute(CExpr, i, i, C_new);

  stmt = stmt.bound(i, i_bounded, (size_t)N, BoundType::MaxExact)
             .split(i_bounded, i0, i1, 32);

  stmt = stmt.concretize();

  stmt = stmt.wsaccel(precomputed, false);
  A.compile(stmt);
  A.assemble();
  A.compute();

  Tensor<double> expected("expected");
  expected() = B(i) * C(i);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, A);
}

TEST(workspaces, tile_dotProduct_3) {
  int N = 1024;
  Tensor<double> A("A");
  Tensor<double> B("B", {N}, Format({Dense}));
  Tensor<double> C("C", {N}, Format({Dense}));

  for (int i = 0; i < N; i++) {
    B.insert({i}, (double) i);
    C.insert({i}, (double) i);
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
  TensorVar B_new("B_new", Type(Float64, {(size_t)N}), taco::dense);
  TensorVar C_new("C_new", Type(Float64, {(size_t)N}), taco::dense);
  TensorVar precomputed("precomputed", Type(Float64, {(size_t)N}), taco::dense);

  stmt = stmt.bound(i, i_bounded, (size_t)N, BoundType::MaxExact)
    .split(i_bounded, i0, i1, 32);
  stmt = stmt.precompute(precomputedExpr, i0, i0, precomputed);

  stmt = stmt.precompute(BExpr, i1, i1, B_new)
    .precompute(CExpr, i1, i1, C_new);


  stmt = stmt.concretize();

  A.compile(stmt);
  A.assemble();
  A.compute();

  Tensor<double> expected("expected");
  expected() = B(i) * C(i);
  expected.compile();
  expected.assemble();
  expected.compute();
  ASSERT_TENSOR_EQ(expected, A);
}
