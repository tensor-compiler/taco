#include "legion.h"
#include "taco_mapper.h"
#include "legion_utils.h"

using namespace Legion;

typedef double valType;

// Defined by the generated TACO code.
void registerTacoTasks();

std::vector<LogicalPartition> partitionForplaceLegionA(Context ctx, Runtime* runtime, LogicalRegion a, int32_t gridX, int32_t gridY);
void placeLegionA(Context ctx, Runtime* runtime, LogicalRegion a, LogicalPartition aPartition, int32_t gridX, int32_t gridY);

std::vector<LogicalPartition> partitionForplaceLegionB(Context ctx, Runtime* runtime, LogicalRegion b, int32_t gridX, int32_t gridY);
void placeLegionB(Context ctx, Runtime* runtime, LogicalRegion b, LogicalPartition bPartition, int32_t gridX, int32_t gridY);

std::vector<LogicalPartition> partitionForplaceLegionC(Context ctx, Runtime* runtime, LogicalRegion c, int32_t gridX, int32_t gridY);
void placeLegionC(Context ctx, Runtime* runtime, LogicalRegion c, LogicalPartition cPartition, int32_t gridX, int32_t gridY);

std::vector<LogicalPartition> partitionForcomputeLegion(Context ctx, Runtime* runtime, LogicalRegion a, LogicalRegion b, LogicalRegion c, int32_t gridX, int32_t gridY);
void computeLegion(Context ctx, Runtime* runtime, LogicalRegion a, LogicalRegion b, LogicalRegion c, LogicalPartition aPartition, LogicalPartition bPartition, LogicalPartition cPartition, int32_t gridX, int32_t gridY);

void top_level_task(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  // Create the regions.
  auto args = runtime->get_input_args();
  int n = -1;
  int gx = -1;
  int gy = -1;
  int px = -1;
  int py = -1;
  // Parse input args.
  for (int i = 1; i < args.argc; i++) {
    if (strcmp(args.argv[i], "-n") == 0) {
      n = atoi(args.argv[++i]);
      continue;
    }
    if (strcmp(args.argv[i], "-gx") == 0) {
      gx = atoi(args.argv[++i]);
      continue;
    }
    if (strcmp(args.argv[i], "-gy") == 0) {
      gy = atoi(args.argv[++i]);
      continue;
    }
    if (strcmp(args.argv[i], "-px") == 0) {
      px = atoi(args.argv[++i]);
      continue;
    }
    if (strcmp(args.argv[i], "-py") == 0) {
      py = atoi(args.argv[++i]);
      continue;
    }
  }
  if (n == -1) {
    std::cout << "Please provide an input matrix size with -n." << std::endl;
    return;
  }
  if (gx == -1) {
    std::cout << "Please provide a grid x size with -gx." << std::endl;
    return;
  }
  if (gy == -1) {
    std::cout << "Please provide a gris y size with -gy." << std::endl;
    return;
  }
  if (px == -1) {
    px = gx;
  }
  if (py == -1) {
    py = gy;
  }

  initCuBLAS(ctx, runtime);

  auto fspace = runtime->create_field_space(ctx);
  allocate_tensor_fields<valType>(ctx, runtime, fspace);
  auto aispace = runtime->create_index_space(ctx, Rect<2>({0, 0}, {n - 1, n - 1}));
  auto bispace = runtime->create_index_space(ctx, Rect<2>({0, 0}, {n - 1, n - 1}));
  auto cispace = runtime->create_index_space(ctx, Rect<2>({0, 0}, {n - 1, n - 1}));
  auto A = runtime->create_logical_region(ctx, aispace, fspace); runtime->attach_name(A, "A");
  auto B = runtime->create_logical_region(ctx, bispace, fspace); runtime->attach_name(B, "B");
  auto C = runtime->create_logical_region(ctx, cispace, fspace); runtime->attach_name(C, "C");


  // Partition all of the tensors.
  // These partitions are disjoint, so we can fill over them.
  auto aPart = partitionForplaceLegionA(ctx, runtime, A, px, py)[0];
  auto bPart = partitionForplaceLegionB(ctx, runtime, B, px, py)[0];
  auto cPart = partitionForplaceLegionC(ctx, runtime, C, px, py)[0];

  // Get partitions for the computation.
  auto parts = partitionForcomputeLegion(ctx, runtime, A, B, C, gx, gy);

  std::vector<size_t> times;
  // Run the benchmark several times.
  for (int i = 0; i < 10; i++) {
    tacoFill<valType>(ctx, runtime, A, aPart, 0);
    tacoFill<valType>(ctx, runtime, B, bPart, 1);
    tacoFill<valType>(ctx, runtime, C, cPart, 1);

    // Place the tensors.
    placeLegionA(ctx, runtime, A, aPart, gx, gy);
    placeLegionB(ctx, runtime, B, bPart, gx, gy);
    placeLegionC(ctx, runtime, C, cPart, gx, gy);

    // Compute on the tensors.
    benchmark(ctx, runtime, times, [&]() { computeLegion(ctx, runtime, A, B, C, parts[0], parts[1], parts[2], gx, gy); });
  }

  // Get the GFLOPS per node.
  auto avgTime = average(times);
  auto flopCount = getGEMMFLOPCount(n, n, n);
  auto gflops = getGFLOPS(flopCount, avgTime);
  auto nodes = runtime->select_tunable_value(ctx, Mapping::DefaultMapper::DEFAULT_TUNABLE_NODE_COUNT).get<size_t>();
  LEGION_PRINT_ONCE(runtime, ctx, stdout, "On %ld nodes achieved GFLOPS per node: %lf.\n", nodes, gflops / double(nodes));

  // The result should be equal to 1.
  tacoValidate<valType>(ctx, runtime, A, aPart, valType(n));
}

TACO_MAIN(valType)
