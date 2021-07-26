#include "legion.h"
#include "taco_mapper.h"
#include "legion_utils.h"

using namespace Legion;

typedef double valType;

// Defined by the generated TACO code.
void registerTacoTasks();
LogicalPartition partitionLegionA(Context ctx, Runtime* runtime, LogicalRegion a, int32_t gx, int32_t gy);
LogicalPartition partitionLegionB(Context ctx, Runtime* runtime, LogicalRegion a, int32_t gx, int32_t gz);
LogicalPartition partitionLegionC(Context ctx, Runtime* runtime, LogicalRegion a, int32_t gz, int32_t gy);
LogicalPartition placeLegionA(Context ctx, Runtime* runtime, LogicalRegion a, int32_t gx, int32_t gy, int32_t gz);
LogicalPartition placeLegionB(Context ctx, Runtime* runtime, LogicalRegion b, int32_t gx, int32_t gy, int32_t gz);
LogicalPartition placeLegionC(Context ctx, Runtime* runtime, LogicalRegion c, int32_t gx, int32_t gy, int32_t gz);
void computeLegion(Context ctx, Runtime* runtime, LogicalRegion a, LogicalRegion b, LogicalRegion c, int32_t gx, int32_t gy, int32_t gz);


void top_level_task(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  // Create the regions.
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
    std::cout << "Please provide an input grid size with -gx." << std::endl;
    return;
  }
  if (gy == -1) {
    std::cout << "Please provide an input grid size with -gy." << std::endl;
    return;
  }
  if (gz == -1) {
    std::cout << "Please provide an input grid size with -gz." << std::endl;
    return;
  }

  auto fspace = runtime->create_field_space(ctx);
  allocate_tensor_fields<valType>(ctx, runtime, fspace);
  auto ispace = runtime->create_index_space(ctx, Rect<2>({0, 0}, {n - 1, n - 1}));
  auto A = runtime->create_logical_region(ctx, ispace, fspace); runtime->attach_name(A, "A");
  auto B = runtime->create_logical_region(ctx, ispace, fspace); runtime->attach_name(B, "B");
  auto C = runtime->create_logical_region(ctx, ispace, fspace); runtime->attach_name(C, "C");

  // Partition all of the tensors.
  auto aPart = partitionLegionA(ctx, runtime, A, gx, gy);
  auto bPart = partitionLegionB(ctx, runtime, B, gx, gz);
  auto cPart = partitionLegionC(ctx, runtime, C, gz, gy);

  std::vector<size_t> times;
  for (int i = 0; i < 10; i++) {
    tacoFill<valType>(ctx, runtime, A, aPart, 0);
    tacoFill<valType>(ctx, runtime, B, bPart, 1);
    tacoFill<valType>(ctx, runtime, C, cPart, 1);

    // Place the tensors.
    placeLegionA(ctx, runtime, A, gx, gy, gz);
    placeLegionB(ctx, runtime, B, gx, gy, gz);
    placeLegionC(ctx, runtime, C, gx, gy, gz);

    // Compute on the tensors.
    benchmark(ctx, runtime, times, [&]() {
      computeLegion(ctx, runtime, A, B, C, gx, gy, gz);
      // Call the placement function again to force reduction along each slice of A.
      placeLegionA(ctx, runtime, A, gx, gy, gz);
    });
  }

  // Get the GFLOPS per node.
  auto avgTime = average(times);
  auto flopCount = getGEMMFLOPCount(n, n, n);
  auto gflops = getGFLOPS(flopCount, avgTime);
  auto nodes = runtime->select_tunable_value(ctx, Mapping::DefaultMapper::DEFAULT_TUNABLE_NODE_COUNT).get<size_t>();
  LEGION_PRINT_ONCE(runtime, ctx, stdout, "On %ld nodes achieved GFLOPS per node: %lf.\n", nodes, gflops / double(nodes));

  tacoValidate<valType>(ctx, runtime, A, valType(n));
}

TACO_MAIN(valType)
