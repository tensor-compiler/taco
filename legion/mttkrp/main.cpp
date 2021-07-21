#include "legion.h"
#include "taco_mapper.h"
#include "legion_utils.h"

using namespace Legion;

typedef double valType;

// Defined by the generated TACO code.
void registerTacoTasks();

// Partitioning statements.
LogicalPartition partitionLegionA(Context ctx, Runtime* runtime, LogicalRegion A, int32_t gridX);
LogicalPartition partitionLegionB(Context ctx, Runtime* runtime, LogicalRegion B, int32_t gridX, int32_t gridY, int32_t gridZ);
LogicalPartition partitionLegionC(Context ctx, Runtime* runtime, LogicalRegion C, int32_t gridY);
LogicalPartition partitionLegionD(Context ctx, Runtime* runtime, LogicalRegion D, int32_t gridZ);

LogicalPartition placeLegionA(Context ctx, Runtime* runtime, LogicalRegion A, int32_t gridX, int32_t gridY, int32_t gridZ);
LogicalPartition placeLegionB(Context ctx, Runtime* runtime, LogicalRegion B, int32_t gridX, int32_t gridY, int32_t gridZ);
LogicalPartition placeLegionC(Context ctx, Runtime* runtime, LogicalRegion C, int32_t gridY, int32_t gridX, int32_t gridZ);
LogicalPartition placeLegionD(Context ctx, Runtime* runtime, LogicalRegion D, int32_t gridZ, int32_t gridX, int32_t gridY);
#ifdef TACO_USE_CUDA
void computeLegion(Context ctx, Runtime* runtime, LogicalRegion A, LogicalRegion B, LogicalRegion C, LogicalRegion D, LogicalPartition BPartition, int32_t gx);
#else
void computeLegion(Context ctx, Runtime* runtime, LogicalRegion A, LogicalRegion B, LogicalRegion C, LogicalRegion D, LogicalPartition BPartition);
#endif

void top_level_task(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  auto args = runtime->get_input_args();
  int n = -1;
  int gx = -1;
  int gy = -1;
  int gz = -1;

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
    if (strcmp(args.argv[i], "-gz") == 0) {
      gz = atoi(args.argv[++i]);
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
  if (gz == -1) {
    std::cout << "Please provide a gris y size with -gy." << std::endl;
    return;
  }

  initCuBLAS(ctx, runtime);

  auto fspace = runtime->create_field_space(ctx);
  allocate_tensor_fields<valType>(ctx, runtime, fspace);
  auto aISpace = runtime->create_index_space(ctx, Rect<2>({0, 0}, {n - 1, n - 1}));
  auto bISpace = runtime->create_index_space(ctx, Rect<3>({0, 0, 0}, {n - 1, n - 1, n - 1}));
  auto cISpace = runtime->create_index_space(ctx, Rect<2>({0, 0}, {n - 1, n - 1}));
  auto dISpace = runtime->create_index_space(ctx, Rect<2>({0, 0}, {n - 1, n - 1}));
  auto A = runtime->create_logical_region(ctx, aISpace, fspace); runtime->attach_name(A, "A");
  auto B = runtime->create_logical_region(ctx, bISpace, fspace); runtime->attach_name(B, "B");
  auto C = runtime->create_logical_region(ctx, cISpace, fspace); runtime->attach_name(C, "C");
  auto D = runtime->create_logical_region(ctx, dISpace, fspace); runtime->attach_name(D, "D");

  // Partition all of the tensors.
  auto aPart = partitionLegionA(ctx, runtime, A, gx);
  auto bPart = partitionLegionB(ctx, runtime, B, gx, gy, gz);
  auto cPart = partitionLegionC(ctx, runtime, C, gy);
  auto dPart = partitionLegionD(ctx, runtime, D, gz);

  std::vector<size_t> times;
  for (int i = 0; i < 11; i++) {
    tacoFill<valType>(ctx, runtime, A, aPart, 0);
    tacoFill<valType>(ctx, runtime, B, bPart, 1);
    tacoFill<valType>(ctx, runtime, C, cPart, 1);
    tacoFill<valType>(ctx, runtime, D, dPart, 1);

    placeLegionA(ctx, runtime, A, gx, gy, gz);
    auto part = placeLegionB(ctx, runtime, B, gx, gy, gz);
    placeLegionC(ctx, runtime, C, gx, gy, gz);
    placeLegionD(ctx, runtime, D, gx, gy, gz);

    auto bench = [&]() {
#ifdef TACO_USE_CUDA
      computeLegion(ctx, runtime, A, B, C, D, part, gx); 
#else
      computeLegion(ctx, runtime, A, B, C, D, part); 
#endif
      // Run the A placement routine again to force a reduction into the right place.
      placeLegionA(ctx, runtime, A, gx, gy, gz);
    };

    if (i == 0) {
      bench();
      tacoValidate<valType>(ctx, runtime, A, aPart, valType(n * n));
    } else {
      benchmark(ctx, runtime, times, bench);
    }
  }

  // Get the GFLOPS per node.
  auto avgTime = average(times);
  auto flopCount = getMTTKRPFLOPCount(n, n, n, n);
  auto gflops = getGFLOPS(flopCount, avgTime);
  auto nodes = runtime->select_tunable_value(ctx, Mapping::DefaultMapper::DEFAULT_TUNABLE_NODE_COUNT).get<size_t>();
  LEGION_PRINT_ONCE(runtime, ctx, stdout, "On %ld nodes achieved GFLOPS per node: %lf.\n", nodes, gflops / double(nodes));
}

TACO_MAIN(valType)
