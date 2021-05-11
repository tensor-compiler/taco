#include "test.h"
#include "taco/tensor.h"
#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/distribution.h"
#include "taco/lower/lower.h"
#include "codegen/codegen.h"
#include "codegen/codegen_legion_c.h"

#include <fstream>

using namespace taco;

TEST(distributed, test) {
  int dim = 10;
  Tensor<int> a("a", {dim}, Format{Dense});
  Tensor<int> b("b", {dim}, Format{Dense});
  Tensor<int> c("c", {dim}, Format{Dense});

  IndexVar i("i"), in("in"), il("il"), il1 ("il1"), il2("il2");
  a(i) = b(i) + c(i);
  auto stmt = a.getAssignment().concretize();
  stmt = stmt.distribute({i}, {in}, {il}, Grid(4));
  stmt = stmt.split(il, il1, il2, 256);


  // Communication modification must go at the end.
  // TODO (rohany): name -- placement
//  stmt = stmt.pushCommUnder(a(i), in).pushCommUnder(b(i), il1);
//  stmt = stmt.pushCommUnder(a(i), il1).pushCommUnder(b(i), il1);
//  stmt = stmt.pushCommUnder(a(i), in).pushCommUnder(b(i), in);
  stmt = stmt.pushCommUnder(a(i), in).pushCommUnder(b(i), in).pushCommUnder(c(i), in);

  auto lowered = lower(stmt, "computeLegion", false, true);
//  std::cout << lowered << std::endl;

  auto codegen = std::make_shared<ir::CodegenLegionC>(std::cout, taco::ir::CodeGen::ImplementationGen);
  codegen->compile(lowered);
}

TEST(distributed, multiDim) {
  int dim = 10;
  Tensor<int> a("a", {dim, dim}, Format{Dense, Dense});
  Tensor<int> b("b", {dim, dim}, Format{Dense, Dense});

  IndexVar i("i"), j("j"), in("in"), jn("jn"), il("il"), jl("jl");
  a(i, j) = b(i, j);
  auto stmt = a.getAssignment().concretize();
  stmt = stmt.distribute({i, j}, {in, jn}, {il, jl}, Grid(4, 4));
  stmt = stmt.pushCommUnder(a(i, j), jn).pushCommUnder(b(i, j), jn);

  auto lowered = lower(stmt, "computeLegion", false, true);
//  std::cout << lowered << std::endl;
  auto codegen = std::make_shared<ir::CodegenLegionC>(std::cout, taco::ir::CodeGen::ImplementationGen);
  codegen->compile(lowered);
}

TEST(distributed, basicComputeOnto) {
  int dim = 10;
//  Tensor<int> a("a", {dim}, Format{Dense});
//  Tensor<int> b("b", {dim}, Format{Dense});
  Tensor<int> a("a", {dim, dim}, Format{Dense, Dense});
  Tensor<int> b("b", {dim, dim}, Format{Dense, Dense});

  IndexVar i("i"), j("j"), in("in"), jn("jn"), il("il"), jl("jl");
//  a(i) = b(i);
  a(i, j) = b(i, j);
  auto stmt = a.getAssignment().concretize();
//  stmt = stmt.distributeOnto({i}, {in}, {il}, a(i));
  stmt = stmt.distributeOnto({i, j}, {in, jn}, {il, jl}, a(i, j));
//  stmt = stmt.pushCommUnder(b(i), in);
  stmt = stmt.pushCommUnder(b(i, j), jn);

  auto lowered = lower(stmt, "computeLegion", false, true);
//  std::cout << lowered << std::endl;
  auto codegen = std::make_shared<ir::CodegenLegionC>(std::cout, taco::ir::CodeGen::ImplementationGen);
  codegen->compile(lowered);
}

TEST(distributed, summaMM) {
  int dim = 10;
  Tensor<int> a("a", {dim, dim}, Format{Dense, Dense});
  Tensor<int> b("b", {dim, dim}, Format{Dense, Dense});
  Tensor<int> c("c", {dim, dim}, Format{Dense, Dense});

  IndexVar i("i"), j("j"), in("in"), jn("jn"), il("il"), jl("jl"), k("k"), ki("ki"), ko("ko");

  a(i, j) = b(i, k) * c(k, j);

  // Place each tensor onto a processor grid.
  auto grid = Grid(2, 2);
  auto placement = GridPlacement({0, 1});
  auto placeA = a.partition(grid).place(grid, placement);
  auto placeB = b.partition(grid).place(grid, placement);
  auto placeC = c.partition(grid).place(grid, placement);
  auto placeALowered = lower(placeA, "placeLegionA", false, true);
  auto placeBLowered = lower(placeB, "placeLegionB", false, true);
  auto placeCLowered = lower(placeC, "placeLegionC", false, true);

  auto stmt = a.getAssignment().concretize();
  stmt = stmt
      .distributeOnto({i, j}, {in, jn}, {il, jl}, a(i, j))
      .split(k, ko, ki, 256)
      .reorder({ko, il, jl})
      .pushCommUnder(b(i, k), ko)
      .pushCommUnder(c(k, j), ko)
      ;

  auto lowered = lower(stmt, "computeLegion", false, true);
  auto codegen = std::make_shared<ir::CodegenLegionC>(std::cout, taco::ir::CodeGen::ImplementationGen);

  // Code-generate all of the placement and compute code.
  auto all = ir::Block::make({placeALowered, placeBLowered, placeCLowered, lowered});
  codegen->compile(all);
  // Also write it into a file.
  {
    ofstream f("../legion/summaMM/taco-generated.cpp");
    auto codegen = std::make_shared<ir::CodegenLegionC>(f, taco::ir::CodeGen::ImplementationGen);
    codegen->compile(all);
    f.close();
  }
}

TEST(distributed, cannonMM) {
  int dim = 10;
  Tensor<int> a("a", {dim, dim}, Format{Dense, Dense});
  Tensor<int> b("b", {dim, dim}, Format{Dense, Dense});
  Tensor<int> c("c", {dim, dim}, Format{Dense, Dense});

  // Place each tensor onto a processor grid.
  auto grid = Grid(2, 2);
  auto placement = GridPlacement({0, 1});
  auto placeA = a.partition(grid).place(grid, placement);
  auto placeB = b.partition(grid).place(grid, placement);
  auto placeC = c.partition(grid).place(grid, placement);
  auto placeALowered = lower(placeA, "placeLegionA", false, true);
  auto placeBLowered = lower(placeB, "placeLegionB", false, true);
  auto placeCLowered = lower(placeC, "placeLegionC", false, true);

  IndexVar i("i"), j("j"), in("in"), jn("jn"), il("il"), jl("jl"), k("k"), ki("ki"), ko("ko"), kos("kos");
  a(i, j) = b(i, k) * c(k, j);
  auto stmt = a.getAssignment().concretize();
  stmt = stmt
      .distributeOnto({i, j}, {in, jn}, {il, jl}, a(i, j))
      .divide(k, ko, ki, 2)
      .reorder({ko, il, jl})
      .stagger(ko, {in, jn}, kos)
      .pushCommUnder(b(i, k), kos)
      .pushCommUnder(c(k, j), kos)
      ;

  auto lowered = lower(stmt, "computeLegion", false, true);
  // std::cout << lowered << std::endl;

  // Code-generate all of the placement and compute code.
  auto all = ir::Block::make({placeALowered, placeBLowered, placeCLowered, lowered});
  auto codegen = std::make_shared<ir::CodegenLegionC>(std::cout, taco::ir::CodeGen::ImplementationGen);
  codegen->compile(all);
  // Also write it into a file.
  {
    ofstream f("../legion/cannonMM/taco-generated.cpp");
    auto codegen = std::make_shared<ir::CodegenLegionC>(f, taco::ir::CodeGen::ImplementationGen);
    codegen->compile(all);
    f.close();
  }
}

TEST(distributed, staggerNoDist) {
  int dim = 10;
  Tensor<int> a("a", {dim, dim}, Format{Dense, Dense});
  Tensor<int> b("b", {dim, dim}, Format{Dense, Dense});
  Tensor<int> expected("expected", {dim, dim}, Format{Dense, Dense});

  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      b.insert({i, j}, 1);
      expected.insert({i, j}, 1);
    }
  }

  IndexVar i("i"), j("j"), js("js");
  a(i, j) = b(i, j);
  auto stmt = a.getAssignment().concretize();
  stmt = stmt.stagger(j, {i}, js);

  a.compile(stmt);
  a.evaluate();
  ASSERT_TRUE(equals(a, expected));
}

TEST(distributed, reduction) {
  int dim = 10;
  Tensor<int> a("a", {dim}, Format{Dense});
  Tensor<int> b("b", {dim, dim}, Format{Dense, Dense});
  Tensor<int> c("c", {dim}, Format{Dense});

  IndexVar i("i"), j("j"), in("in"), jn("jn"), il("il"), jl("jl");

  a(i) = b(i, j) * c(j);
  auto stmt = a.getAssignment().concretize();
  stmt = stmt
      .distribute({i, j}, {in, jn}, {il, jl}, Grid(2, 2))
      .pushCommUnder(a(i), jn)
      .pushCommUnder(b(i, j), jn)
      .pushCommUnder(c(j), jn)
      ;

  auto lowered = lower(stmt, "computeLegion", false, true);
//  std::cout << lowered << std::endl;
  auto codegen = std::make_shared<ir::CodegenLegionC>(std::cout, taco::ir::CodeGen::ImplementationGen);
  codegen->compile(lowered);
}

TEST(distributed, packingPlacement) {
  int dim = 10;
  Tensor<int> a("a", {dim, dim}, Format{Dense, Dense});
  auto grid = Grid(4, 4);
  auto placeGrid = Grid(4, 4, 4);
  auto stmt = a.partition(grid).place(placeGrid, GridPlacement({0, 1, Face(0)}));
  auto lowered = lower(stmt, "placeLegion", false, true);
  auto codegen = std::make_shared<ir::CodegenLegionC>(std::cout, taco::ir::CodeGen::ImplementationGen);
  codegen->compile(lowered);
  // Also write it into a file.
  {
    ofstream f("../legion/placement-test/taco-generated.cpp");
    auto codegen = std::make_shared<ir::CodegenLegionC>(f, taco::ir::CodeGen::ImplementationGen);
    codegen->compile(lowered);
    f.close();
  }
}

TEST(distributed, placement) {
  int dim = 10;

  auto toString = [](IndexStmt stmt) {
    std::stringstream ss;
    ss << stmt;
    return ss.str();
  };

  {
    // Simple partitioning of a vector onto a vector of processors.
    Tensor<int> a("a", {dim}, Format{Dense});
    auto grid = Grid(4);
    a.partition(grid);
    auto stmt = a.place(grid, GridPlacement({0}));
    ASSERT_EQ(toString(stmt), "suchthat(forall(in, forall(il, place(a(i))), Distributed, ParallelReduction, transfers: transfer(a(i))), divide(i, in, il, 4))");
  }
  {
    // Place a matrix onto a vector of processors.
    Tensor<int> a("a", {dim, dim}, Format{Dense, Dense});
    // TODO (rohany): Here is a good place to test partitioning the other dimension of the matrix.
    auto grid = Grid(4);
    auto placeGrid = Grid(4);
    a.partition(grid);
    auto stmt = a.place(placeGrid, GridPlacement({0}));
    ASSERT_EQ(toString(stmt), "suchthat(forall(in, forall(il, forall(j, place(a(i,j)))), Distributed, ParallelReduction, transfers: transfer(a(i,j))), divide(i, in, il, 4))");
    // TODO (rohany): It seems like doing GridPlacement({1}) mimics partitioning across the y axis.
  }
  {
    // Place a vector onto a grid in different ways.
    Tensor<int> a("a", {dim}, Format{Dense});
    auto grid = Grid(4);
    auto placeGrid = Grid(4, 4);
    a.partition(grid);
    // Place the vector so that each row of the processor grid holds the chunk of the vector.
    ASSERT_EQ(toString(a.place(placeGrid, GridPlacement({0, Replicate()}))), "suchthat(forall(distFused, forall(il, forall(jl, place(a(i)))), Distributed, ParallelReduction, transfers: transfer(a(i))), divide(i, in, il, 4) and divide(j, jn, jl, 4) and multiFuse({in, jn}, reorder(in, jn)))");
    // Place the vector so that each column of the processor grid holds the chunk of the vector.
    ASSERT_EQ(toString(a.place(placeGrid, GridPlacement({Replicate(), 0}))), "suchthat(forall(distFused, forall(il, forall(jl, place(a(j)))), Distributed, ParallelReduction, transfers: transfer(a(j))), divide(i, in, il, 4) and divide(j, jn, jl, 4) and multiFuse({in, jn}, reorder(in, jn)))");
  }
  {
    // Place a matrix onto a 3-dimensional grid in different ways.
    Tensor<int> a("a", {dim, dim}, Format{Dense, Dense});
    auto grid = Grid(4, 4);
    auto placeGrid = Grid(4, 4, 4);
    a.partition(grid);
    // Replicate the tensor along each dimension in turn.
    ASSERT_EQ(toString(a.place(placeGrid, GridPlacement({0, 1, Replicate()}))), "suchthat(forall(distFused, forall(il, forall(jl, forall(kl, place(a(i,j))))), Distributed, ParallelReduction, transfers: transfer(a(i,j))), divide(i, in, il, 4) and divide(j, jn, jl, 4) and divide(k, kn, kl, 4) and multiFuse({in, jn, kn}, reorder(in, jn, kn)))");
    ASSERT_EQ(toString(a.place(placeGrid, GridPlacement({0, Replicate(), 1}))), "suchthat(forall(distFused, forall(il, forall(jl, forall(kl, place(a(i,k))))), Distributed, ParallelReduction, transfers: transfer(a(i,k))), divide(i, in, il, 4) and divide(j, jn, jl, 4) and divide(k, kn, kl, 4) and multiFuse({in, jn, kn}, reorder(in, jn, kn)))");
    ASSERT_EQ(toString(a.place(placeGrid, GridPlacement({Replicate(), 0, 1}))), "suchthat(forall(distFused, forall(il, forall(jl, forall(kl, place(a(j,k))))), Distributed, ParallelReduction, transfers: transfer(a(j,k))), divide(i, in, il, 4) and divide(j, jn, jl, 4) and divide(k, kn, kl, 4) and multiFuse({in, jn, kn}, reorder(in, jn, kn)))");
    // Placing the tensor in different orientations (like put the columns along the first axis of the grid).
    ASSERT_EQ(toString(a.place(placeGrid, GridPlacement({1, 0, Replicate()}))), "suchthat(forall(distFused, forall(il, forall(jl, forall(kl, place(a(j,i))))), Distributed, ParallelReduction, transfers: transfer(a(j,i))), divide(i, in, il, 4) and divide(j, jn, jl, 4) and divide(k, kn, kl, 4) and multiFuse({in, jn, kn}, reorder(in, jn, kn)))");
    ASSERT_EQ(toString(a.place(placeGrid, GridPlacement({1, Replicate(), 0}))), "suchthat(forall(distFused, forall(il, forall(jl, forall(kl, place(a(k,i))))), Distributed, ParallelReduction, transfers: transfer(a(k,i))), divide(i, in, il, 4) and divide(j, jn, jl, 4) and divide(k, kn, kl, 4) and multiFuse({in, jn, kn}, reorder(in, jn, kn)))");
    ASSERT_EQ(toString(a.place(placeGrid, GridPlacement({Replicate(), 1, 0}))), "suchthat(forall(distFused, forall(il, forall(jl, forall(kl, place(a(k,j))))), Distributed, ParallelReduction, transfers: transfer(a(k,j))), divide(i, in, il, 4) and divide(j, jn, jl, 4) and divide(k, kn, kl, 4) and multiFuse({in, jn, kn}, reorder(in, jn, kn)))");
  }
}

TEST(distributed, placement2) {
  int dim = 10;
//  Tensor<int> a("a", {dim}, Format{Dense});
//  auto grid = Grid(4);
//  auto placeGrid = Grid(4, 4);
//  a.partition(grid);
//  auto stmt = a.place(placeGrid, GridPlacement({0, Replicate()}));
  Tensor<int> a("a", {dim, dim}, Format{Dense, Dense});
  auto grid = Grid(3, 3);
  auto placeGrid = Grid(3, 3, 3);
  a.partition(grid);
  // Replicate the tensor along each dimension in turn.
//  auto stmt = a.place(placeGrid, GridPlacement({0, Replicate(), 1}));
  auto stmt = a.place(placeGrid, GridPlacement({Replicate(), 0, 1}));
  auto lowered = lower(stmt, "computeLegion", false, true);
  auto codegen = std::make_shared<ir::CodegenLegionC>(std::cout, taco::ir::CodeGen::ImplementationGen);
  codegen->compile(lowered);
}

TEST(distributed, nesting) {
  int dim = 10;
  Tensor<int> a("a", {dim, dim}, Format{Dense, Dense});
  Tensor<int> b("b", {dim, dim}, Format{Dense, Dense});

  IndexVar i("i"), j("j");
  a(i, j) = b(j, i);
  auto stmt = a.getAssignment().concretize();
  auto lowered = lower(stmt, "computeLegion", false, true);
  std::cout << lowered << std::endl;
}