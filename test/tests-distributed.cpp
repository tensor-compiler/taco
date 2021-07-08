#include "test.h"
#include "taco/tensor.h"
#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/distribution.h"
#include "taco/lower/lower.h"
#include "codegen/codegen.h"
#include "codegen/codegen_legion_c.h"
#include "codegen/codegen_legion_cuda.h"
#include "codegen/codegen_cuda.h"

#include "taco/index_notation/transformations.h"
#include "taco/index_notation/provenance_graph.h"

#include "taco/ir/simplify.h"

#include <fstream>

using namespace taco;

const int NNZ_PER_THREAD=8;
const int WARP_SIZE = 32;
const int BLOCK_SIZE=256;
const int NNZ_PER_WARP = NNZ_PER_THREAD * WARP_SIZE;
const int NNZ_PER_TB = NNZ_PER_THREAD * BLOCK_SIZE;

TEST(distributed, test) {
  int dim = 10;
  Tensor<int> a("a", {dim}, Format{Dense});
  Tensor<int> b("b", {dim}, Format{Dense});
  Tensor<int> c("c", {dim}, Format{Dense});

  IndexVar i("i"), in("in"), il("il"), il1 ("il1"), il2("il2");
  a(i) = b(i) + c(i);
  auto stmt = a.getAssignment().concretize();
  stmt = stmt.distribute({i}, {in}, {il}, Grid(4));

  // Communication modification must go at the end.
  // TODO (rohany): name -- placement
//  stmt = stmt.pushCommUnder(a(i), in).communicate(b(i), il1);
//  stmt = stmt.pushCommUnder(a(i), il1).communicate(b(i), il1);
//  stmt = stmt.pushCommUnder(a(i), in).communicate(b(i), in);
  stmt = stmt.communicate(a(i), in).communicate(b(i), in).communicate(c(i), in);

  auto lowered = lower(stmt, "computeLegion", false, true);
//  std::cout << lowered << std::endl;
  auto codegen = std::make_shared<ir::CodegenLegionC>(std::cout, taco::ir::CodeGen::ImplementationGen);
  codegen->compile(lowered);
}

TEST(distributed, cuda_test) {
  int dim = 10;
  Tensor<int> a("a", {dim}, Format{Dense});
  Tensor<int> b("b", {dim}, Format{Dense});
  IndexVar i("i"), in("in"), il("il"), il1 ("il1"), il2("il2");
  a(i) = b(i);
  auto stmt = a.getAssignment().concretize();
  stmt = stmt.distribute({i}, {in}, {il}, Grid(4));
  int NNZ_PER_THREAD=8;
  int WARP_SIZE = 32;
  int BLOCK_SIZE=256;
  int NNZ_PER_WARP = NNZ_PER_THREAD * WARP_SIZE;
  int NNZ_PER_TB = NNZ_PER_THREAD * BLOCK_SIZE;
  IndexVar f1("f1"), f2("f2"), f3("f3"), f4("f4"), block("bvar"), warp("wvar"), thread("tvar");
  stmt = stmt.split(il, block, f1, NNZ_PER_TB)
      .split(f1, warp, f2, NNZ_PER_WARP)
      .split(f2, thread, f3, NNZ_PER_THREAD)
      .parallelize(block, ParallelUnit::GPUBlock, taco::OutputRaceStrategy::IgnoreRaces)
      .parallelize(warp, ParallelUnit::GPUWarp, taco::OutputRaceStrategy::IgnoreRaces)
      .parallelize(thread, ParallelUnit::GPUThread, taco::OutputRaceStrategy::IgnoreRaces)
      ;
  stmt = stmt.communicate(a(i), in).communicate(b(i), in);
  auto lowered = lower(stmt, "computeLegion", false, true);
  auto codegen = std::make_shared<ir::CodegenLegionCuda>(std::cout, taco::ir::CodeGen::ImplementationGen);
  codegen->compile(lowered);
  {
    ofstream f("../legion/cuda-test/taco-generated.cu");
    auto codegen = std::make_shared<ir::CodegenLegionCuda>(f, taco::ir::CodeGen::ImplementationGen);
    codegen->compile(lowered);
    f.close();
  }
}

TEST(distributed, multiDim) {
  int dim = 10;
  Tensor<int> a("a", {dim, dim}, Format{Dense, Dense});
  Tensor<int> b("b", {dim, dim}, Format{Dense, Dense});

  IndexVar i("i"), j("j"), in("in"), jn("jn"), il("il"), jl("jl");
  a(i, j) = b(i, j);
  auto stmt = a.getAssignment().concretize();
  stmt = stmt.distribute({i, j}, {in, jn}, {il, jl}, Grid(4, 4));
  stmt = stmt.communicate(a(i, j), jn).communicate(b(i, j), jn);

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
//  stmt = stmt.distribute({i}, {in}, {il}, a(i));
  stmt = stmt.distribute({i, j}, {in, jn}, {il, jl}, a(i, j));
//  stmt = stmt.communicate(b(i), in);
  stmt = stmt.communicate(b(i, j), jn);

  auto lowered = lower(stmt, "computeLegion", false, true);
//  std::cout << lowered << std::endl;
  auto codegen = std::make_shared<ir::CodegenLegionC>(std::cout, taco::ir::CodeGen::ImplementationGen);
  codegen->compile(lowered);
}

TEST(distributed, summaMM) {
  int dim = 10;
  Tensor<double> a("a", {dim, dim}, Format{Dense, Dense});
  Tensor<double> b("b", {dim, dim}, Format{Dense, Dense});
  Tensor<double> c("c", {dim, dim}, Format{Dense, Dense});

  IndexVar i("i"), j("j"), in("in"), jn("jn"), il("il"), jl("jl"), k("k"), ki("ki"), ko("ko");
  IndexVar iln("iln"), ill("ill");
  a(i, j) = b(i, k) * c(k, j);

  // Place each tensor onto a processor grid.
  auto gx = ir::Var::make("gridX", Int32, false, false, true);
  auto gy = ir::Var::make("gridY", Int32, false, false, true);
  auto grid = Grid(gx, gy);
  auto partitionLowered = lower(a.partitionStmt(grid), "partitionLegion", false, true);
  auto placement = GridPlacement({0, 1});
  auto placeA = a.partition(grid).place(grid, placement);
  auto placeB = b.partition(grid).place(grid, placement);
  auto placeC = c.partition(grid).place(grid, placement);
  auto placeALowered = lower(placeA, "placeLegionA", false, true);
  auto placeBLowered = lower(placeB, "placeLegionB", false, true);
  auto placeCLowered = lower(placeC, "placeLegionC", false, true);

  std::shared_ptr<LeafCallInterface> gemm = std::make_shared<GEMM>();
  auto stmt = a.getAssignment().concretize();
  stmt = stmt
      .distribute({i, j}, {in, jn}, {il, jl}, a(i, j))
      .divide(k, ko, ki, gx)
      .reorder({ko, il, jl})
      .communicate(b(i, k), ko)
      .communicate(c(k, j), ko)
      // Hierarchically parallelize the computation for each NUMA region.
      // TODO (rohany): Make the number of OpenMP processors configurable.
      .distribute({il}, {iln}, {ill}, Grid(2))
      .communicate(a(i, j), iln)
      .communicate(b(i, k), iln)
      .communicate(c(k, j), iln)
      .swapLeafKernel(ill, gemm)
      ;

  auto lowered = lower(stmt, "computeLegion", false, true);
  auto codegen = std::make_shared<ir::CodegenLegionC>(std::cout, taco::ir::CodeGen::ImplementationGen);

  // Code-generate all of the placement and compute code.
  auto all = ir::Block::make({partitionLowered, placeALowered, placeBLowered, placeCLowered, lowered});
  codegen->compile(all);
  // Also write it into a file.
  {
    ofstream f("../legion/summaMM/taco-generated.cpp");
    auto codegen = std::make_shared<ir::CodegenLegionC>(f, taco::ir::CodeGen::ImplementationGen);
    codegen->compile(all);
    f.close();
  }
}

TEST(distributed, cuda_summaMM) {
  int dim = 10;

  // Place each tensor onto a processor grid.
  auto gx = ir::Var::make("gridX", Int32, false, false, true);
  auto gy = ir::Var::make("gridY", Int32, false, false, true);
  auto grid = Grid(gx, gy);
  auto partGrid = Grid(ir::Mul::make(gx, 2), ir::Mul::make(gy, 2));
  auto placement = GridPlacement({0, 1});

  auto gpuGrid = Grid(2, 2);
  std::vector<TensorDistribution> dist{
    TensorDistribution(grid),
    TensorDistribution(gpuGrid, taco::ParallelUnit::DistributedGPU),
  };

  Tensor<double> a("a", {dim, dim}, Format{Dense, Dense}, dist);
  Tensor<double> b("b", {dim, dim}, Format{Dense, Dense}, dist);
  Tensor<double> c("c", {dim, dim}, Format{Dense, Dense}, dist);

  auto nodePart = lower(a.partitionStmt(grid), "partitionLegionNode", false, true);
  auto partitionLowered = lower(a.partitionStmt(partGrid), "partitionLegion", false, true);
  auto placeALowered = lower(a.getPlacementStatement(), "placeLegionA", false, true);
  auto placeBLowered = lower(b.getPlacementStatement(), "placeLegionB", false, true);
  auto placeCLowered = lower(c.getPlacementStatement(), "placeLegionC", false, true);
  IndexVar i("i"), j("j"), in("in"), jn("jn"), il("il"), jl("jl"), k("k"), ki("ki"), ko("ko"), kos("kos");
  IndexVar iln("iln"), ill("ill"), jln("jln"), jll("jll"), kii("kii"), kio("kio"), kios("kios");
  std::shared_ptr<LeafCallInterface> gemm = std::make_shared<CuGEMM>();
  a(i, j) = b(i, k) * c(k, j);
  auto stmt = a.getAssignment().concretize();
  stmt = stmt
      // Schedule for each node.
      .distribute({i, j}, {in, jn}, {il, jl}, a(i, j))
      .divide(k, ko, ki, gx)
      .reorder({ko, il, jl})
      .communicate(b(i, k), ko)
      .communicate(c(k, j), ko)
      // Schedule for each GPU within a node.
      .distribute({il, jl}, {iln, jln}, {ill, jll}, Grid(2, 2), taco::ParallelUnit::DistributedGPU)
      .divide(ki, kio, kii, 2)
      .reorder({kio, ill, jll})
      .communicate(b(i, k), kio)
      .communicate(c(k, j), kio)
      .communicate(a(i, j), jln)
      .swapLeafKernel(ill, gemm)
      ;
  auto lowered = lower(stmt, "computeLegion", false, true);
  auto all = ir::Block::make({partitionLowered, placeALowered, placeBLowered, placeCLowered, lowered});
  ofstream f("../legion/summaMM/taco-generated.cu");
  auto codegen = std::make_shared<ir::CodegenLegionCuda>(f, taco::ir::CodeGen::ImplementationGen);
  codegen->compile(all);
  f.close();
}

TEST(distributed, singlemttkrp) {
  int dim = 1000;
  Tensor<double> A("A", {dim, dim}, Dense);
  Tensor<double> B("B", {dim, dim, dim}, {Dense, Dense, Dense});
  Tensor<double> C("C", {dim, dim}, Dense);
  Tensor<double> D("D", {dim, dim}, Dense);

  IndexVar i("i"), j("j"), k("k"), l("l");
  IndexVar io("io"), ii("ii");
  A(i, l) = B(i, j, k) * C(j, l) * D(k, l);
  // This schedule generates the code found in `leaf_kernels.h`.
  auto stmt = A.getAssignment().concretize()
               .reorder({i, j, k, l})
               .split(i, io, ii, 4)
               .parallelize(ii, taco::ParallelUnit::CPUVector, taco::OutputRaceStrategy::NoRaces)
               .parallelize(io, taco::ParallelUnit::CPUThread, taco::OutputRaceStrategy::NoRaces)
               ;
  A.compile(stmt);
  std::cout << A.getSource() << std::endl;
}

TEST(distributed, mttkrp) {
  int dim = 1000;

  // Implementing the algorithm from https://par.nsf.gov/servlets/purl/10078535.
  auto gx = ir::Var::make("gridX", Int32, false, false, true);
  auto gy = ir::Var::make("gridY", Int32, false, false, true);
  auto gz = ir::Var::make("gridZ", Int32, false, false, true);
  auto grid3 = Grid(gx, gy, gz);

  // Partition and place the 3-tensor onto a 3-d grid.
  TensorDistribution BDist(grid3);
  // Partition the matrices in a single dimension, and place them along different
  // edges of the processor grid.
  TensorDistribution ADist(
    Grid(gx),
    grid3,
    // We want this along the edge that the `i` dimension of B is partitioned along.
    GridPlacement({0, Face(0), Face(0)})
  );
  TensorDistribution CDist(
    Grid(gy),
    grid3,
    // We want this along the edge that the `j` dimension of B is partitioned along.
    GridPlacement({Face(0), 0, Face(0)})
  );
  TensorDistribution DDist(
    Grid(gz),
    grid3,
    // We want this along the edge that the `j` dimension of B is partitioned along.
    GridPlacement({Face(0), Face(0), 0})
  );

  Tensor<double> A("A", {dim, dim}, {Dense, Dense}, ADist);
  Tensor<double> B("B", {dim, dim, dim}, {Dense, Dense, Dense}, BDist);
  Tensor<double> C("C", {dim, dim}, {Dense, Dense}, CDist);
  Tensor<double> D("D", {dim, dim}, {Dense, Dense}, DDist);

  std::shared_ptr<LeafCallInterface> mttkrp = std::make_shared<MTTKRP>();
  IndexVar i("i"), j("j"), k("k"), l("l");
  IndexVar in("in"), il("il"), jn("jn"), jl("jl"), kn("kn"), kl("kl");
  IndexVar ii("ii"), io("io");
  A(i, l) = B(i, j, k) * C(j, l) * D(k, l);
  auto stmt = A.getAssignment().concretize()
               .reorder({i, j, k, l})
               .distribute({i, j, k}, {in, jn, kn}, {il, jl, kl}, B(i, j, k))
               .reorder({il, jl, kl, l})
               .communicate(A(i, l), kn)
               .communicate(C(j, l), kn)
               .communicate(D(k, l), kn)
               .swapLeafKernel(il, mttkrp)
               ;

  // Generate partitioning statements for each tensor.
  auto partitionA = lower(A.partitionStmt(Grid(gx)), "partitionLegionA", false, true);
  auto partitionB = lower(B.partitionStmt(grid3), "partitionLegionB", false, true);
  auto partitionC = lower(C.partitionStmt(Grid(gy)), "partitionLegionC", false, true);
  auto partitionD = lower(D.partitionStmt(Grid(gz)), "partitionLegionD", false, true);

  // Placement statements for each tensor.
  auto placeALowered = lower(A.getPlacementStatement(), "placeLegionA", false, true);
  auto placeBLowered = lower(B.getPlacementStatement(), "placeLegionB", false, true);
  auto placeCLowered = lower(C.getPlacementStatement(), "placeLegionC", false, true);
  auto placeDLowered = lower(D.getPlacementStatement(), "placeLegionD", false, true);
  auto lowered = lower(stmt, "computeLegion", false, true);
  auto all = ir::Block::make({
    partitionA, partitionB, partitionC, partitionD,
    placeALowered, placeBLowered, placeCLowered, placeDLowered,
    lowered
  });
  auto codegen = std::make_shared<ir::CodegenLegionC>(std::cout, taco::ir::CodeGen::ImplementationGen);
  codegen->compile(all);
  // Also write it into a file.
  {
    ofstream f("../legion/mttkrp/taco-generated.cpp");
    auto codegen = std::make_shared<ir::CodegenLegionC>(f, taco::ir::CodeGen::ImplementationGen);
    codegen->compile(all);
    f.close();
  }
}

TEST(distributed, cuda_mttkrp) {
  int dim = 1000;

  // Implementing the algorithm from https://par.nsf.gov/servlets/purl/10078535.
  auto gx = ir::Var::make("gridX", Int32, false, false, true);
  auto gy = ir::Var::make("gridY", Int32, false, false, true);
  auto gz = ir::Var::make("gridZ", Int32, false, false, true);
  auto grid3 = Grid(gx, gy, gz);

  // Partition and place the 3-tensor onto a 3-d grid.
  TensorDistribution BDist(grid3, taco::ParallelUnit::DistributedGPU);
  // Partition the matrices in a single dimension, and place them along different
  // edges of the processor grid.
  TensorDistribution ADist(
      Grid(gx),
      grid3,
      // We want this along the edge that the `i` dimension of B is partitioned along.
      GridPlacement({0, Face(0), Face(0)}),
      taco::ParallelUnit::DistributedGPU
  );
  TensorDistribution CDist(
      Grid(gy),
      grid3,
      // We want this along the edge that the `j` dimension of B is partitioned along.
      GridPlacement({Face(0), 0, Face(0)}),
      taco::ParallelUnit::DistributedGPU
  );
  TensorDistribution DDist(
      Grid(gz),
      grid3,
      // We want this along the edge that the `j` dimension of B is partitioned along.
      GridPlacement({Face(0), Face(0), 0}),
      taco::ParallelUnit::DistributedGPU
  );

  Tensor<double> A("A", {dim, dim}, {Dense, Dense}, ADist);
  Tensor<double> B("B", {dim, dim, dim}, {Dense, Dense, Dense}, BDist);
  Tensor<double> C("C", {dim, dim}, {Dense, Dense}, CDist);
  Tensor<double> D("D", {dim, dim}, {Dense, Dense}, DDist);

  std::shared_ptr<LeafCallInterface> mttkrp = std::make_shared<CuMTTKRP>();
  IndexVar i("i"), j("j"), k("k"), l("l");
  IndexVar in("in"), il("il"), jn("jn"), jl("jl"), kn("kn"), kl("kl");
  IndexVar ii("ii"), io("io");
  A(i, l) = B(i, j, k) * C(j, l) * D(k, l);
  // Since heirarchical reductions don't work yet, we'll start with a flat implementation.
  auto stmt = A.getAssignment().concretize()
      .reorder({i, j, k, l})
      .distribute({i, j, k}, {in, jn, kn}, {il, jl, kl}, B(i, j, k), taco::ParallelUnit::DistributedGPU)
      .reorder({il, jl, kl, l})
      .communicate(A(i, l), kn)
      .communicate(C(j, l), kn)
      .communicate(D(k, l), kn)
      .swapLeafKernel(il, mttkrp)
  ;

  // Generate partitioning statements for each tensor.
  auto partitionA = lower(A.partitionStmt(Grid(gx)), "partitionLegionA", false, true);
  auto partitionB = lower(B.partitionStmt(grid3), "partitionLegionB", false, true);
  auto partitionC = lower(C.partitionStmt(Grid(gy)), "partitionLegionC", false, true);
  auto partitionD = lower(D.partitionStmt(Grid(gz)), "partitionLegionD", false, true);

  // Placement statements for each tensor.
  auto placeALowered = lower(A.getPlacementStatement(), "placeLegionA", false, true);
  auto placeBLowered = lower(B.getPlacementStatement(), "placeLegionB", false, true);
  auto placeCLowered = lower(C.getPlacementStatement(), "placeLegionC", false, true);
  auto placeDLowered = lower(D.getPlacementStatement(), "placeLegionD", false, true);
  auto lowered = lower(stmt, "computeLegion", false, true);
  auto all = ir::Block::make({
                                 partitionA, partitionB, partitionC, partitionD,
                                 placeALowered, placeBLowered, placeCLowered, placeDLowered,
                                 lowered
                             });
  auto codegen = std::make_shared<ir::CodegenLegionCuda>(std::cout, taco::ir::CodeGen::ImplementationGen);
  codegen->compile(all);
  // Also write it into a file.
  {
    ofstream f("../legion/mttkrp/taco-generated.cu");
    auto codegen = std::make_shared<ir::CodegenLegionCuda>(f, taco::ir::CodeGen::ImplementationGen);
    codegen->compile(all);
    f.close();
  }
}

TEST(distributed, ttv) {
  int dim = 1000;

  // Our implementation of TTV will block partition A and B, and then replicate
  // C onto each block.
  auto gx = ir::Var::make("gridX", Int32, false, false, true);
  auto gy = ir::Var::make("gridY", Int32, false, false, true);
  auto grid = Grid(gx, gy);
  TensorDistribution distribution(grid);
  // Prereplicate C onto all of the nodes.
  TensorDistribution cDistribution(
    Grid(),
    grid,
    GridPlacement({Replicate(), Replicate()})
  );
  Tensor<double> A("A", {dim, dim}, {Dense, Dense}, distribution);
  Tensor<double> B("B", {dim, dim, dim}, {Dense, Dense, Dense}, distribution);
  Tensor<double> C("C", {dim}, {Dense}, cDistribution);

  auto partitionA = lower(A.partitionStmt(grid), "partitionLegionA", false, true);
  auto partitionB = lower(B.partitionStmt(grid), "partitionLegionB", false, true);

  auto placeALowered = lower(A.getPlacementStatement(), "placeLegionA", false, true);
  auto placeBLowered = lower(B.getPlacementStatement(), "placeLegionB", false, true);
  auto placeCLowered = lower(C.getPlacementStatement(), "placeLegionC", false, true);

  IndexVar i("i"), j("j"), k("k"), l("l");
  IndexVar in("in"), il("il"), jn("jn"), jl("jl");
  IndexVar ii("ii"), io("io");
  A(i, j) = B(i, j, k) * C(k);
  auto stmt = A.getAssignment().concretize()
               .distribute({i, j}, {in, jn}, {il, jl}, B(i, j, k))
               .communicate(A(i, j), jn)
               .communicate(C(k), jn)
               .reorder({il, jl, k})
               .split(il, ii, io, 4)
               .parallelize(io, taco::ParallelUnit::CPUVector, taco::OutputRaceStrategy::NoRaces)
               .parallelize(ii, taco::ParallelUnit::CPUThread, taco::OutputRaceStrategy::NoRaces)
               ;
  auto lowered = lower(stmt, "computeLegion", false, true);
  auto all = ir::Block::make({partitionA, partitionB, placeALowered, placeBLowered, placeCLowered, lowered});
  auto codegen = std::make_shared<ir::CodegenLegionC>(std::cout, taco::ir::CodeGen::ImplementationGen);
  codegen->compile(all);
  // Also write it into a file.
  {
    ofstream f("../legion/ttv/taco-generated.cpp");
    auto codegen = std::make_shared<ir::CodegenLegionC>(f, taco::ir::CodeGen::ImplementationGen);
    codegen->compile(all);
    f.close();
  }
}

TEST(distributed, cuda_ttv) {
  int dim = 1000;

  // Our implementation of TTV will block partition A and B, and then replicate
  // C onto each block.
  auto gx = ir::Var::make("gridX", Int32, false, false, true);
  auto gy = ir::Var::make("gridY", Int32, false, false, true);
  auto grid = Grid(gx, gy);
  TensorDistribution distribution(grid, taco::ParallelUnit::DistributedGPU);
  // Prereplicate C onto all of the nodes.
  TensorDistribution cDistribution(
      Grid(),
      grid,
      GridPlacement({Replicate(), Replicate()}),
      taco::ParallelUnit::DistributedGPU
  );
  Tensor<double> A("A", {dim, dim}, {Dense, Dense}, distribution);
  Tensor<double> B("B", {dim, dim, dim}, {Dense, Dense, Dense}, distribution);
  Tensor<double> C("C", {dim}, {Dense}, cDistribution);

  auto partitionA = lower(A.partitionStmt(grid), "partitionLegionA", false, true);
  auto partitionB = lower(B.partitionStmt(grid), "partitionLegionB", false, true);

  auto placeALowered = lower(A.getPlacementStatement(), "placeLegionA", false, true);
  auto placeBLowered = lower(B.getPlacementStatement(), "placeLegionB", false, true);
  auto placeCLowered = lower(C.getPlacementStatement(), "placeLegionC", false, true);

  IndexVar i("i"), j("j"), k("k"), l("l");
  IndexVar in("in"), il("il"), jn("jn"), jl("jl");
  IndexVar ii("ii"), io("io"), f("f");
  A(i, j) = B(i, j, k) * C(k);
  auto stmt = A.getAssignment().concretize()
      // TODO (rohany): We could do distributeOnto here once the bug regarding bounds checks
      //  for it is fixed.
      // .distribute({i, j}, {in, jn}, {il, jl}, B(i, j, k), taco::ParallelUnit::DistributedGPU)
      .distribute({i, j}, {in, jn}, {il, jl}, grid, taco::ParallelUnit::DistributedGPU)
      .communicate(A(i, j), jn)
      .communicate(B(i, j, k), jn)
      .communicate(C(k), jn)
      .fuse(il, jl, f)
      .split(f, io, ii, 64)
      .parallelize(io, taco::ParallelUnit::GPUBlock, taco::OutputRaceStrategy::NoRaces)
      .parallelize(ii, taco::ParallelUnit::GPUThread, taco::OutputRaceStrategy::NoRaces)
      ;

  auto lowered = lower(stmt, "computeLegion", false, true);
  auto all = ir::Block::make({partitionA, partitionB, placeALowered, placeBLowered, placeCLowered, lowered});
  auto codegen = std::make_shared<ir::CodegenLegionCuda>(std::cout, taco::ir::CodeGen::ImplementationGen);
  codegen->compile(all);
  // Also write it into a file.
  {
    ofstream f("../legion/ttv/taco-generated.cu");
    auto codegen = std::make_shared<ir::CodegenLegionCuda>(f, taco::ir::CodeGen::ImplementationGen);
    codegen->compile(all);
    f.close();
  }
}

TEST(distributed, ttmc) {
  int dim = 1000;
  auto pieces = ir::Var::make("pieces", Int32, false, false, true);
  auto grid = Grid(pieces);
  TensorDistribution dist(grid);
  TensorDistribution repl(
    Grid(),
    grid,
    GridPlacement({Replicate()})
  );

  Tensor<double> A("A", {dim, dim, dim}, {Dense, Dense, Dense}, dist);
  Tensor<double> B("B", {dim, dim, dim}, {Dense, Dense, Dense}, dist);
  Tensor<double> C("C", {dim, dim}, {Dense, Dense}, repl);

  IndexVar i("i"), j("j"), k("k"), l("l"), m("m");
  IndexVar in("in"), il("il");
  IndexVar ii("ii"), io("io"), ji("ji"), jo("jo"), li("li"), lo("lo"), lii("lii"), lio("lio");

  std::shared_ptr<LeafCallInterface> ttmc = std::make_shared<TTMC>();
  A(i, j, l) = B(i, j, k) * C(k, l);
  auto stmt = A.getAssignment().concretize()
               .distribute({i}, {in}, {il}, grid)
               .communicate(A(i, j, l), in)
               .communicate(B(i, j, k), in)
               .communicate(C(k, l), in)
               .swapLeafKernel(il, ttmc)
               ;

  auto partition3tensor = lower(A.partitionStmt(grid), "partition3Tensor", false, true);
  auto placeALowered = lower(A.getPlacementStatement(), "placeLegionA", false, true);
  auto placeBLowered = lower(B.getPlacementStatement(), "placeLegionB", false, true);
  auto placeCLowered = lower(C.getPlacementStatement(), "placeLegionC", false, true);
  auto lowered = lower(stmt, "computeLegion", false, true);
  auto all = ir::Block::make({partition3tensor, placeALowered, placeBLowered, placeCLowered, lowered});
  auto codegen = std::make_shared<ir::CodegenLegionC>(std::cout, taco::ir::CodeGen::ImplementationGen);
  codegen->compile(all);
  // Also write it into a file.
  {
    ofstream f("../legion/ttmc/taco-generated.cpp");
    auto codegen = std::make_shared<ir::CodegenLegionC>(f, taco::ir::CodeGen::ImplementationGen);
    codegen->compile(all);
    f.close();
  }
}

TEST(distributed, cuda_ttmc) {
  int dim = 1000;
  auto pieces = ir::Var::make("pieces", Int32, false, false, true);
  auto grid = Grid(pieces);
  TensorDistribution dist(grid, taco::ParallelUnit::DistributedGPU);
  TensorDistribution repl(
      Grid(),
      grid,
      GridPlacement({Replicate()}),
      taco::ParallelUnit::DistributedGPU
  );

  Tensor<double> A("A", {dim, dim, dim}, {Dense, Dense, Dense}, dist);
  Tensor<double> B("B", {dim, dim, dim}, {Dense, Dense, Dense}, dist);
  Tensor<double> C("C", {dim, dim}, {Dense, Dense}, repl);

  IndexVar i("i"), j("j"), k("k"), l("l"), m("m");
  IndexVar in("in"), il("il");
  IndexVar ii("ii"), io("io"), ji("ji"), jo("jo"), li("li"), lo("lo"), lii("lii"), lio("lio");

  std::shared_ptr<LeafCallInterface> ttmc = std::make_shared<CuTTMC>();
  A(i, j, l) = B(i, j, k) * C(k, l);
  auto stmt = A.getAssignment().concretize()
      .distribute({i}, {in}, {il}, grid, taco::ParallelUnit::DistributedGPU)
      .communicate(A(i, j, l), in)
      .communicate(B(i, j, k), in)
      .communicate(C(k, l), in)
      .swapLeafKernel(il, ttmc)
      ;
  auto partition3tensor = lower(A.partitionStmt(grid), "partition3Tensor", false, true);
  auto placeALowered = lower(A.getPlacementStatement(), "placeLegionA", false, true);
  auto placeBLowered = lower(B.getPlacementStatement(), "placeLegionB", false, true);
  auto placeCLowered = lower(C.getPlacementStatement(), "placeLegionC", false, true);
  auto lowered = lower(stmt, "computeLegion", false, true);
  auto all = ir::Block::make({partition3tensor, placeALowered, placeBLowered, placeCLowered, lowered});
  auto codegen = std::make_shared<ir::CodegenLegionCuda>(std::cout, taco::ir::CodeGen::ImplementationGen);
  codegen->compile(all);
  // Also write it into a file.
  {
    ofstream f("../legion/ttmc/taco-generated.cu");
    auto codegen = std::make_shared<ir::CodegenLegionCuda>(f, taco::ir::CodeGen::ImplementationGen);
    codegen->compile(all);
    f.close();
  }
}

TEST(distributed, cannonMM) {
  int dim = 10;
  // Place each tensor onto a processor grid.
  auto gx = ir::Var::make("gridX", Int32, false, false, true);
  auto gy = ir::Var::make("gridY", Int32, false, false, true);
  auto grid = Grid(gx, gy);
  auto placement = GridPlacement({0, 1});

  TensorDistribution distribution(grid);
  Tensor<double> a("a", {dim, dim}, Format{Dense, Dense}, distribution);
  Tensor<double> b("b", {dim, dim}, Format{Dense, Dense}, distribution);
  Tensor<double> c("c", {dim, dim}, Format{Dense, Dense}, distribution);

  auto partitionLowered = lower(a.partitionStmt(grid), "partitionLegion", false, true);
  auto placeALowered = lower(a.getPlacementStatement(), "placeLegionA", false, true);
  auto placeBLowered = lower(b.getPlacementStatement(), "placeLegionB", false, true);
  auto placeCLowered = lower(c.getPlacementStatement(), "placeLegionC", false, true);

  IndexVar i("i"), j("j"), in("in"), jn("jn"), il("il"), jl("jl"), k("k"), ki("ki"), ko("ko"), kos("kos");
  IndexVar iln("iln"), ill("ill");
  std::shared_ptr<LeafCallInterface> gemm = std::make_shared<GEMM>();
  a(i, j) = b(i, k) * c(k, j);
  auto stmt = a.getAssignment().concretize();
  stmt = stmt
      .distribute({i, j}, {in, jn}, {il, jl}, a(i, j))
      .divide(k, ko, ki, gx)
      .reorder({ko, il, jl})
      .stagger(ko, {in, jn}, kos)
      .communicate(b(i, k), kos)
      .communicate(c(k, j), kos)
      // Hierarchically parallelize the computation for each NUMA region.
      // TODO (rohany): Make the number of OpenMP processors configurable.
      .distribute({il}, {iln}, {ill}, Grid(2))
      .communicate(a(i, j), iln)
      .communicate(b(i, k), iln)
      .communicate(c(k, j), iln)
      .swapLeafKernel(ill, gemm)
      ;

  auto lowered = lower(stmt, "computeLegion", false, true);
  // Code-generate all of the placement and compute code.
  auto all = ir::Block::make({partitionLowered, placeALowered, placeBLowered, placeCLowered, lowered});
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

TEST(distributed, cuda_cannonMM) {
  int dim = 10;

  // Place each tensor onto a processor grid.
  auto gx = ir::Var::make("gridX", Int32, false, false, true);
  auto gy = ir::Var::make("gridY", Int32, false, false, true);
  auto grid = Grid(gx, gy);
  auto partGrid = Grid(ir::Mul::make(gx, 2), ir::Mul::make(gy, 2));
  auto placement = GridPlacement({0, 1});

  auto gpuGrid = Grid(2, 2);
  std::vector<TensorDistribution> dist{
    TensorDistribution(grid, ParallelUnit::DistributedNode),
    TensorDistribution(gpuGrid, ParallelUnit::DistributedGPU),
  };

  Tensor<double> a("a", {dim, dim}, Format{Dense, Dense}, dist);
  Tensor<double> b("b", {dim, dim}, Format{Dense, Dense}, dist);
  Tensor<double> c("c", {dim, dim}, Format{Dense, Dense}, dist);

  auto nodePart = lower(a.partitionStmt(grid), "partitionLegionNode", false, true);
  auto partitionLowered = lower(a.partitionStmt(partGrid), "partitionLegion", false, true);
  auto placeALowered = lower(a.getPlacementStatement(), "placeLegionA", false, true);
  auto placeBLowered = lower(b.getPlacementStatement(), "placeLegionB", false, true);
  auto placeCLowered = lower(c.getPlacementStatement(), "placeLegionC", false, true);

  IndexVar i("i"), j("j"), in("in"), jn("jn"), il("il"), jl("jl"), k("k"), ki("ki"), ko("ko"), kos("kos");
  IndexVar iln("iln"), ill("ill"), jln("jln"), jll("jll"), kii("kii"), kio("kio"), kios("kios");
  std::shared_ptr<LeafCallInterface> gemm = std::make_shared<CuGEMM>();
  a(i, j) = b(i, k) * c(k, j);
  auto stmt = a.getAssignment().concretize();
  stmt = stmt
      // Schedule for each node.
      .distribute({i, j}, {in, jn}, {il, jl}, grid, taco::ParallelUnit::DistributedNode)
      .divide(k, ko, ki, gx)
      .reorder({ko, il, jl})
      .stagger(ko, {in, jn}, kos)
      .communicate(a(i, j), in)
      .communicate(b(i, k), kos)
      .communicate(c(k, j), kos)
      // Schedule for each GPU within a node.
      .distribute({il, jl}, {iln, jln}, {ill, jll}, Grid(2, 2), taco::ParallelUnit::DistributedGPU)
      .divide(ki, kio, kii, 2)
      .reorder({kio, ill, jll})
      .stagger(kio, {iln, jln}, kios)
      .communicate(b(i, k), kios)
      .communicate(c(k, j), kios)
      .communicate(a(i, j), jln)
      .swapLeafKernel(ill, gemm)
      ;
  auto lowered = lower(stmt, "computeLegion", false, true);
  auto all = ir::Block::make({partitionLowered, placeALowered, placeBLowered, placeCLowered, lowered});
  ofstream f("../legion/cannonMM/taco-generated.cu");
  auto codegen = std::make_shared<ir::CodegenLegionCuda>(f, taco::ir::CodeGen::ImplementationGen);
  codegen->compile(all);
  f.close();
}

TEST(distributed, johnsonMM) {
  int dim = 10;
  Tensor<double> a("a", {dim, dim}, Format{Dense, Dense});
  Tensor<double> b("b", {dim, dim}, Format{Dense, Dense});
  Tensor<double> c("c", {dim, dim}, Format{Dense, Dense});

  // Each tensor lives on a different face of the processor cube.
  auto gdim = ir::Var::make("gridDim", Int32, false, false, true);
  auto grid = Grid(gdim, gdim);
  auto cube = Grid(gdim, gdim, gdim);
  auto partitionLowered = lower(a.partitionStmt(grid), "partitionLegion", false, true);
  auto placeA = a.partition(grid).place(cube, GridPlacement({0, 1, Face(0)}));
  auto placeB = b.partition(grid).place(cube, GridPlacement({0, Face(0), 1}));
  auto placeC = c.partition(grid).place(cube, GridPlacement({Face(0), 0, 1}));
  auto placeALowered = lower(placeA, "placeLegionA", false, true);
  auto placeBLowered = lower(placeB, "placeLegionB", false, true);
  auto placeCLowered = lower(placeC, "placeLegionC", false, true);

  IndexVar i("i"), j("j"), k("k"), in("in"), il("il"), jn("jn"), jl("jl"), kn("kn"), kl("kl");
  IndexVar iln("iln"), ill("ill");
  a(i, j) = b(i, k) * c(k, j);
  auto stmt = a.getAssignment().concretize();
  std::shared_ptr<LeafCallInterface> gemm = std::make_shared<GEMM>();
  stmt = stmt
      .distribute({i, j, k}, {in, jn, kn}, {il, jl, kl}, cube)
      .communicate(a(i, j), kn)
      .communicate(b(i, k), kn)
      .communicate(c(k, j), kn)
      // Hierarchically parallelize the computation for each NUMA region.
      // TODO (rohany): Make the number of OpenMP processors configurable.
      .distribute({il}, {iln}, {ill}, Grid(2))
      .communicate(a(i, j), iln)
      .communicate(b(i, k), iln)
      .communicate(c(k, j), iln)
      .swapLeafKernel(ill, gemm)
      ;
  auto lowered = lowerNoWait(stmt, "computeLegion");
  // Code-generate all of the placement and compute code.
  auto all = ir::Block::make({partitionLowered, placeALowered, placeBLowered, placeCLowered, lowered});
  auto codegen = std::make_shared<ir::CodegenLegionC>(std::cout, taco::ir::CodeGen::ImplementationGen);
  codegen->compile(all);
  // Also write it into a file.
  {
    ofstream f("../legion/johnsonMM/taco-generated.cpp");
    auto codegen = std::make_shared<ir::CodegenLegionC>(f, taco::ir::CodeGen::ImplementationGen);
    codegen->compile(all);
    f.close();
  }
}

TEST(distributed, cuda_johnsonMM) {
  int dim = 10;
  auto gdim = ir::Var::make("gridDim", Int32, false, false, true);
  auto grid = Grid(gdim, gdim);
  auto cube = Grid(gdim, gdim, gdim);
  auto gpuDim = ir::Var::make("gpuDim", Int32, false, false, true);
  auto gpuGrid = Grid(gpuDim, gpuDim);
  TensorDistribution gpuDist(gpuGrid, taco::ParallelUnit::DistributedGPU);
  std::vector<TensorDistribution> aDist{
    TensorDistribution{
      grid,
      cube,
      GridPlacement({0, 1, Face(0)}),
    },
    gpuDist,
  };
  std::vector<TensorDistribution> bDist{
    TensorDistribution{
      grid,
      cube,
      GridPlacement({0, Face(0), 1}),
    },
    gpuDist,
  };
  std::vector<TensorDistribution> cDist{
    TensorDistribution{
      grid,
      cube,
      GridPlacement({Face(0), 0, 1}),
    },
    gpuDist,
  };
  Tensor<double> a("a", {dim, dim}, Format{Dense, Dense}, aDist);
  Tensor<double> b("b", {dim, dim}, Format{Dense, Dense}, bDist);
  Tensor<double> c("c", {dim, dim}, Format{Dense, Dense}, cDist);

  auto partDim = ir::Mul::make(gdim, gpuDim);
  auto partitionStmt = lower(a.partitionStmt(Grid(partDim, partDim)), "partitionLegion", false, true);
  auto placeALowered = lower(a.getPlacementStatement(), "placeLegionA", false, true);
  auto placeBLowered = lower(b.getPlacementStatement(), "placeLegionB", false, true);
  auto placeCLowered = lower(c.getPlacementStatement(), "placeLegionC", false, true);

  IndexVar i("i"), j("j"), k("k"), in("in"), il("il"), jn("jn"), jl("jl"), kn("kn"), kl("kl");
  IndexVar iln("iln"), ill("ill"), jln("jln"), jll("jll"), kii("kii"), kio("kio"), kios("kios");
  a(i, j) = b(i, k) * c(k, j);
  auto stmt = a.getAssignment().concretize();
  std::shared_ptr<LeafCallInterface> gemm = std::make_shared<CuGEMM>();
  stmt = stmt
      .distribute({i, j, k}, {in, jn, kn}, {il, jl, kl}, cube)
      .communicate(a(i, j), kn)
      .communicate(b(i, k), kn)
      .communicate(c(k, j), kn)
      .distribute({il, jl}, {iln, jln}, {ill, jll}, gpuGrid, taco::ParallelUnit::DistributedGPU)
      .divide(kl, kio, kii, gpuDim)
      .reorder({kio, ill, jll})
      .stagger(kio, {iln, jln}, kios)
      .communicate(b(i, k), kios)
      .communicate(c(k, j), kios)
      .communicate(a(i, j), jln)
      .swapLeafKernel(ill, gemm)
      ;

  auto lowered = lower(stmt, "computeLegion", false, true);
  auto all = ir::Block::make({partitionStmt, placeALowered, placeBLowered, placeCLowered, lowered});
  auto codegen = std::make_shared<ir::CodegenLegionCuda>(std::cout, taco::ir::CodeGen::ImplementationGen);
  codegen->compile(all);
  {
    ofstream f("../legion/johnsonMM/taco-generated.cu");
    auto codegen = std::make_shared<ir::CodegenLegionCuda>(f, taco::ir::CodeGen::ImplementationGen);
    codegen->compile(all);
    f.close();
  }
}

TEST(distributed, solomonikMM) {
  int dim = 10;
  Tensor<int> a("a", {dim, dim}, Format{Dense, Dense});
  Tensor<int> b("b", {dim, dim}, Format{Dense, Dense});
  Tensor<int> c("c", {dim, dim}, Format{Dense, Dense});

  int procs = 64;
  int C = 2;
  int rpoc = sqrt(procs / C);
  int rpoc3 = sqrt(procs / (pow(C, 3)));

  // All tensors are distributed onto the i-j face of the process cube.
  Grid partGrid = Grid(rpoc, rpoc);
  Grid procGrid = Grid(rpoc, rpoc, C);
  auto placeA = a.partition(partGrid).place(procGrid, GridPlacement({0, 1, Face(0)}));
  auto placeB = b.partition(partGrid).place(procGrid, GridPlacement({0, 1, Face(0)}));
  auto placeC = c.partition(partGrid).place(procGrid, GridPlacement({0, 1, Face(0)}));
  auto placeALowered = lower(placeA, "placeLegionA", false, true);
  auto placeBLowered = lower(placeB, "placeLegionB", false, true);
  auto placeCLowered = lower(placeC, "placeLegionC", false, true);

  IndexVar i("i"), j("j"), k("k"), in("in"), il("il"), jn("jn"), jl("jl"), kn("kn"), kl("kl"), k1("k1"), k2("k2"), k1s("k1s");
  a(i, j) = b(i, k) * c(k, j);
  auto stmt = a.getAssignment().concretize();
  // To schedule for solomonik's algorithm, we'll distribute over i, j, k according to the
  // processor grid. Then, we divide the kl loop into k1 and k2 so that each partition of C
  // is operated on in chunks. Finally, we then stagger the k1 loop so that along each parallel
  // slice of k, a Cannon style shifting occurs.
  stmt = stmt
      .distribute({i, j, k}, {in, jn, kn}, {il, jl, kl}, procGrid)
      .divide(kl, k1, k2, rpoc3)
      .reorder({k1, il, jl})
      .stagger(k1, {in, jn}, k1s)
      .communicate(a(i, j), jn)
      .communicate(b(i, k), k1s)
      .communicate(c(k, j), k1s)
      ;
  auto lowered = lower(stmt, "computeLegion", false, true);
  // Code-generate all of the placement and compute code.
  auto all = ir::Block::make({placeALowered, placeBLowered, placeCLowered, lowered});
  auto codegen = std::make_shared<ir::CodegenLegionC>(std::cout, taco::ir::CodeGen::ImplementationGen);
  codegen->compile(all);
  {
    ofstream f("../legion/solomonikMM/taco-generated.cpp");
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
      .communicate(a(i), jn)
      .communicate(b(i, j), jn)
      .communicate(c(j), jn)
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

TEST(distributed, heirPlacement) {
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
    auto stmt = a.placeHierarchy({
      std::tuple<Grid,Grid,GridPlacement,ParallelUnit>{grid, grid, GridPlacement({0}), ParallelUnit::DistributedNode},
      std::tuple<Grid,Grid,GridPlacement,ParallelUnit>{grid, grid, GridPlacement({0}), ParallelUnit::DistributedNode},
    });
    ASSERT_EQ(toString(stmt), "suchthat(forall(in, forall(iln, forall(ill, place(a(i))), Distributed, ParallelReduction, transfers: transfer(a(i))), Distributed, ParallelReduction, transfers: transfer(a(i))), divide(i, in, il, 4) and divide(il, iln, ill, 4))");
  }
  {
    // Doubly partition a matrix into matrices on each sub-partition.
    Tensor<int> a("a", {dim, dim}, Format{Dense, Dense});
    auto grid = Grid(4, 4);
    auto stmt = a.placeHierarchy({
      std::tuple<Grid,Grid,GridPlacement,ParallelUnit>{grid, grid, GridPlacement({0, 1}), ParallelUnit::DistributedNode},
      std::tuple<Grid,Grid,GridPlacement,ParallelUnit>{grid, grid, GridPlacement({0, 1}), ParallelUnit::DistributedNode},
    });
    ASSERT_EQ(toString(stmt), "suchthat(forall(distFused, forall(distFused1, forall(ill, forall(jll, place(a(i,j)))), Distributed, ParallelReduction, transfers: transfer(a(i,j))), Distributed, ParallelReduction, transfers: transfer(a(i,j))), divide(i, in, il, 4) and divide(j, jn, jl, 4) and multiFuse({in, jn}, reorder(in, jn)) and divide(il, iln, ill, 4) and divide(jl, jln, jll, 4) and multiFuse({iln, jln}, reorder(iln, jln)))");
  }
  {
    Tensor<int> a("a", {dim, dim}, Format{Dense, Dense});
    auto g1 = Grid(4);
    auto g2 = Grid(4, 4);
    auto stmt = a.placeHierarchy({
        std::tuple<Grid,Grid,GridPlacement,ParallelUnit>{g2, g2, GridPlacement({0, Replicate()}), ParallelUnit::DistributedNode},
        std::tuple<Grid,Grid,GridPlacement,ParallelUnit>{g2, g2, GridPlacement({0, 1}), ParallelUnit::DistributedNode},
    });
    ASSERT_EQ(toString(stmt), "suchthat(forall(distFused, forall(distFused1, forall(ill, forall(jl, forall(kl, place(a(i,j))))), Distributed, ParallelReduction, transfers: transfer(a(i,j))), Distributed, ParallelReduction, transfers: transfer(a(i,j))), divide(i, in, il, 4) and divide(k, kn, kl, 4) and multiFuse({in, kn}, reorder(in, kn)) and divide(il, iln, ill, 4) and divide(j, jn, jl, 4) and multiFuse({iln, jn}, reorder(iln, jn)))");
  }
  {
    Tensor<int> a("a", {dim, dim}, Format{Dense, Dense});
    auto grid = Grid(4, 4);
    auto stmt = a.placeHierarchy({
        std::tuple<Grid,Grid,GridPlacement,ParallelUnit>{grid, grid, GridPlacement({0, 1}), ParallelUnit::DistributedNode},
        std::tuple<Grid,Grid,GridPlacement,ParallelUnit>{grid, grid, GridPlacement({0, 1}), ParallelUnit::DistributedGPU},
    });
    std::cout << stmt << std::endl;
    auto lowered = lower(stmt, "placeLegion", false, true);
    auto codegen = std::make_shared<ir::CodegenLegionCuda>(std::cout, taco::ir::CodeGen::ImplementationGen);
    codegen->compile(lowered);
  }
  {
    Tensor<int> a("a", {dim, dim}, Format{Dense, Dense});
    auto grid = Grid(4, 4);
    auto pgrid = Grid(4, 4, 4);
    auto stmt = a.placeHierarchy({
        std::tuple<Grid,Grid,GridPlacement,ParallelUnit>{grid, pgrid, GridPlacement({0, 1, Face(0)}), ParallelUnit::DistributedNode},
        std::tuple<Grid,Grid,GridPlacement,ParallelUnit>{grid, pgrid, GridPlacement({0, 1, Face(0)}), ParallelUnit::DistributedGPU},
    });
    std::cout << stmt << std::endl;
    auto lowered = lower(stmt, "placeLegion", false, true);
    auto codegen = std::make_shared<ir::CodegenLegionCuda>(std::cout, taco::ir::CodeGen::ImplementationGen);
    codegen->compile(lowered);
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
    ASSERT_EQ(toString(a.place(placeGrid, GridPlacement({Replicate(), 0}))), "suchthat(forall(distFused, forall(jl, forall(il, place(a(i)))), Distributed, ParallelReduction, transfers: transfer(a(i))), divide(j, jn, jl, 4) and divide(i, in, il, 4) and multiFuse({jn, in}, reorder(jn, in)))");
  }
  {
    // Place a matrix onto a 3-dimensional grid in different ways.
    Tensor<int> a("a", {dim, dim}, Format{Dense, Dense});
    auto grid = Grid(4, 4);
    auto placeGrid = Grid(4, 4, 4);
    a.partition(grid);
    // Replicate the tensor along each dimension in turn.
    ASSERT_EQ(toString(a.place(placeGrid, GridPlacement({0, 1, Replicate()}))), "suchthat(forall(distFused, forall(il, forall(jl, forall(kl, place(a(i,j))))), Distributed, ParallelReduction, transfers: transfer(a(i,j))), divide(i, in, il, 4) and divide(j, jn, jl, 4) and divide(k, kn, kl, 4) and multiFuse({in, jn, kn}, reorder(in, jn, kn)))");
    ASSERT_EQ(toString(a.place(placeGrid, GridPlacement({0, Replicate(), 1}))), "suchthat(forall(distFused, forall(il, forall(kl, forall(jl, place(a(i,j))))), Distributed, ParallelReduction, transfers: transfer(a(i,j))), divide(i, in, il, 4) and divide(k, kn, kl, 4) and divide(j, jn, jl, 4) and multiFuse({in, kn, jn}, reorder(in, kn, jn)))");
    ASSERT_EQ(toString(a.place(placeGrid, GridPlacement({Replicate(), 0, 1}))), "suchthat(forall(distFused, forall(kl, forall(il, forall(jl, place(a(i,j))))), Distributed, ParallelReduction, transfers: transfer(a(i,j))), divide(k, kn, kl, 4) and divide(i, in, il, 4) and divide(j, jn, jl, 4) and multiFuse({kn, in, jn}, reorder(kn, in, jn)))");
    // Placing the tensor in different orientations (like put the columns along the first axis of the grid).
    ASSERT_EQ(toString(a.place(placeGrid, GridPlacement({1, 0, Replicate()}))), "suchthat(forall(distFused, forall(jl, forall(il, forall(kl, place(a(i,j))))), Distributed, ParallelReduction, transfers: transfer(a(i,j))), divide(j, jn, jl, 4) and divide(i, in, il, 4) and divide(k, kn, kl, 4) and multiFuse({jn, in, kn}, reorder(jn, in, kn)))");
    ASSERT_EQ(toString(a.place(placeGrid, GridPlacement({1, Replicate(), 0}))), "suchthat(forall(distFused, forall(jl, forall(kl, forall(il, place(a(i,j))))), Distributed, ParallelReduction, transfers: transfer(a(i,j))), divide(j, jn, jl, 4) and divide(k, kn, kl, 4) and divide(i, in, il, 4) and multiFuse({jn, kn, in}, reorder(jn, kn, in)))");
    ASSERT_EQ(toString(a.place(placeGrid, GridPlacement({Replicate(), 1, 0}))), "suchthat(forall(distFused, forall(kl, forall(jl, forall(il, place(a(i,j))))), Distributed, ParallelReduction, transfers: transfer(a(i,j))), divide(k, kn, kl, 4) and divide(j, jn, jl, 4) and divide(i, in, il, 4) and multiFuse({kn, jn, in}, reorder(kn, jn, in)))");
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