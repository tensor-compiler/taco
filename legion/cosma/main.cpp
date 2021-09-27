#include "legion.h"
#include "taco_mapper.h"
#include "legion_utils.h"

using namespace Legion;

typedef double valType;

// Defined by the generated TACO code.
void registerTacoTasks();

std::vector<LogicalPartition> partitionForplaceLegionA(Context ctx, Runtime* runtime, LogicalRegion a, int32_t gx, int32_t gy);
void placeLegionA(Context ctx, Runtime* runtime, LogicalRegion a, LogicalPartition aPartition, int32_t gx, int32_t gy, Legion::PrivilegeMode priv, int32_t gz);

std::vector<LogicalPartition> partitionForplaceLegionB(Context ctx, Runtime* runtime, LogicalRegion b, int32_t gx, int32_t gz);
void placeLegionB(Context ctx, Runtime* runtime, LogicalRegion b, LogicalPartition bPartition, int32_t gx, int32_t gz, Legion::PrivilegeMode priv, int32_t gy);

std::vector<LogicalPartition> partitionForplaceLegionC(Context ctx, Runtime* runtime, LogicalRegion c, int32_t gy, int32_t gz);
void placeLegionC(Context ctx, Runtime* runtime, LogicalRegion c, LogicalPartition cPartition, int32_t gy, int32_t gz, Legion::PrivilegeMode priv, int32_t gx);

std::vector<LogicalPartition> partitionForcomputeLegion(Context ctx, Runtime* runtime, LogicalRegion a, LogicalRegion b, LogicalRegion c, int32_t gx, int32_t gy, int32_t gz);
void computeLegion(Context ctx, Runtime* runtime, LogicalRegion a, LogicalRegion b, LogicalRegion c, LogicalPartition aPartition, LogicalPartition bPartition, LogicalPartition cPartition, int32_t gx, int32_t gy, int32_t gz);

void top_level_task(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  // Create the regions.
  auto args = runtime->get_input_args();
  int n = -1;
  int gx = -1;
  int gy = -1;
  int gz = -1;
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
    if (strcmp(args.argv[i], "-gz") == 0) {
      gz = atoi(args.argv[++i]);
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

  initCuBLAS(ctx, runtime);

  auto fspace = runtime->create_field_space(ctx);
  allocate_tensor_fields<valType>(ctx, runtime, fspace);
  auto ispace = runtime->create_index_space(ctx, Rect<2>({0, 0}, {n - 1, n - 1}));
  auto A = runtime->create_logical_region(ctx, ispace, fspace); runtime->attach_name(A, "A");
  auto B = runtime->create_logical_region(ctx, ispace, fspace); runtime->attach_name(B, "B");
  auto C = runtime->create_logical_region(ctx, ispace, fspace); runtime->attach_name(C, "C");

  // Partition all of the tensors.
  auto aPart = partitionForplaceLegionA(ctx, runtime, A, gx, gy)[0];
  auto bPart = partitionForplaceLegionB(ctx, runtime, B, gx, gz)[0];
  auto cPart = partitionForplaceLegionC(ctx, runtime, C, gy, gz)[0];

  // Partition for the computation.
  auto parts = partitionForcomputeLegion(ctx, runtime, A, B, C, gx, gy, gz);

  std::vector<size_t> times;
  for (int i = 0; i < 11; i++) {
    // tacoFill<valType>(ctx, runtime, A, aPart, 0);
    // tacoFill<valType>(ctx, runtime, B, bPart, 1);
    // tacoFill<valType>(ctx, runtime, C, cPart, 1);

    // Place the tensors. Similar to Johnson's algorithm, we let the placement
    // operation also create the valid instances of the data.
    placeLegionA(ctx, runtime, A, aPart, gx, gy, WRITE_ONLY, gz);
    placeLegionB(ctx, runtime, B, bPart, gx, gz, WRITE_ONLY, gy);
    placeLegionC(ctx, runtime, C, cPart, gy, gz, WRITE_ONLY, gx);

    auto bench = [&]() {
      computeLegion(ctx, runtime, A, B, C, parts[0], parts[1], parts[2], gx, gy, gz);
      // Call the placement function again to force reduction along each slice of A.
      placeLegionA(ctx, runtime, A, aPart, gx, gy, READ_ONLY, gz);
    };

    // Compute on the tensors.
    if (i == 0) {
      bench();
    } else {
      benchmark(ctx, runtime, times, bench);
    }
  }

  // Get the GFLOPS per node.
  auto avgTime = average(times);
  auto flopCount = getGEMMFLOPCount(n, n, n);
  auto gflops = getGFLOPS(flopCount, avgTime);
  auto nodes = runtime->select_tunable_value(ctx, Mapping::DefaultMapper::DEFAULT_TUNABLE_NODE_COUNT).get<size_t>();
  LEGION_PRINT_ONCE(runtime, ctx, stdout, "On %ld nodes achieved GFLOPS per node: %lf.\n", nodes, gflops / double(nodes));

  // tacoValidate<valType>(ctx, runtime, A, valType(n));
}

TACO_MAIN(valType)
