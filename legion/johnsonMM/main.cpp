#include "legion.h"
#include "taco_mapper.h"
#include "legion_utils.h"

using namespace Legion;

typedef double valType;

// Defined by the generated TACO code.
void registerTacoTasks();

std::vector<LogicalPartition> partitionForplaceLegionA(Context ctx, Runtime* runtime, LogicalRegion a, int32_t gridDim);
void placeLegionA(Context ctx, Runtime* runtime, LogicalRegion a, LogicalPartition aPartition, int32_t gridDim, Legion::PrivilegeMode priv);

std::vector<LogicalPartition> partitionForplaceLegionB(Context ctx, Runtime* runtime, LogicalRegion b, int32_t gridDim);
void placeLegionB(Context ctx, Runtime* runtime, LogicalRegion b, LogicalPartition bPartition, int32_t gridDim, Legion::PrivilegeMode priv);

std::vector<LogicalPartition> partitionForplaceLegionC(Context ctx, Runtime* runtime, LogicalRegion c, int32_t gridDim);
void placeLegionC(Context ctx, Runtime* runtime, LogicalRegion c, LogicalPartition cPartition, int32_t gridDim, Legion::PrivilegeMode priv);

std::vector<LogicalPartition> partitionForcomputeLegion(Context ctx, Runtime* runtime, LogicalRegion a, LogicalRegion b, LogicalRegion c, int32_t gridDim);
void computeLegion(Context ctx, Runtime* runtime, LogicalRegion a, LogicalRegion b, LogicalRegion c, LogicalPartition aPartition, LogicalPartition bPartition, LogicalPartition cPartition, int32_t gridDim);

void top_level_task(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  // Create the regions.
  auto args = runtime->get_input_args();
  int n = -1;
  int gdim = -1;
  // Parse input args.
  for (int i = 1; i < args.argc; i++) {
    if (strcmp(args.argv[i], "-n") == 0) {
      n = atoi(args.argv[++i]);
      continue;
    }
    if (strcmp(args.argv[i], "-gdim") == 0) {
      gdim = atoi(args.argv[++i]);
      continue;
    }
  }
  if (n == -1) {
    std::cout << "Please provide an input matrix size with -n." << std::endl;
    return;
  }
  if (gdim == -1) {
    std::cout << "Please provide an input grid size with -gdim." << std::endl;
    return;
  }

  initCuBLAS(ctx, runtime);

  auto fspace = runtime->create_field_space(ctx);
  allocate_tensor_fields<valType>(ctx, runtime, fspace);
  auto ispace = runtime->create_index_space(ctx, Rect<2>({0, 0}, {n - 1, n - 1}));
  auto A = runtime->create_logical_region(ctx, ispace, fspace); runtime->attach_name(A, "A");
  auto B = runtime->create_logical_region(ctx, ispace, fspace); runtime->attach_name(B, "B");
  auto C = runtime->create_logical_region(ctx, ispace, fspace); runtime->attach_name(C, "C");

  // Partitions for the placement statements.
  auto paPart = partitionForplaceLegionA(ctx, runtime, A, gdim)[0];
  auto pbPart = partitionForplaceLegionB(ctx, runtime, B, gdim)[0];
  auto pcPart = partitionForplaceLegionC(ctx, runtime, C, gdim)[0];

  // Partitions for the compute statements.
  auto parts = partitionForcomputeLegion(ctx, runtime, A, B, C, gdim);

  std::vector<size_t> times;
  for (int i = 0; i < 11; i++) {
    // We don't currently fill these tensors as the fill isn't "distribution aware", and results
    // in creating some extra instances that we don't want. Instead, we'll just use the placement
    // operation as a dummy "fill" and not look at the result, because it's probably full of garbage.
    // tacoFill<valType>(ctx, runtime, A, paPart, 0);
    // tacoFill<valType>(ctx, runtime, B, pbPart, 1);
    // tacoFill<valType>(ctx, runtime, C, pcPart, 1);

    // Place the tensors. We use the WRITE_ONLY privilege to ensure that these placed instances
    // are valid instances to read from.
    placeLegionA(ctx, runtime, A, paPart, gdim, WRITE_ONLY);
    placeLegionB(ctx, runtime, B, pbPart, gdim, WRITE_ONLY);
    placeLegionC(ctx, runtime, C, pcPart, gdim, WRITE_ONLY);

    auto bench = [&]() {
      computeLegion(ctx, runtime, A, B, C, parts[0], parts[1], parts[2], gdim);
      // Call the placement function again to force reduction along each slice of A.
      placeLegionA(ctx, runtime, A, paPart, gdim, READ_ONLY);
    };

    // Run one iteration of the algorithm to warm up the system.
    if (i == 0) {
      bench();
    } else {
      // Compute on the tensors.
      benchmark(ctx, runtime, times, bench);
    }
  }

  // Get the GFLOPS per node.
  auto avgTime = average(times);
  auto flopCount = getGEMMFLOPCount(n, n, n);
  auto gflops = getGFLOPS(flopCount, avgTime);
  auto nodes = runtime->select_tunable_value(ctx, Mapping::DefaultMapper::DEFAULT_TUNABLE_NODE_COUNT).get<size_t>();
  LEGION_PRINT_ONCE(runtime, ctx, stdout, "On %ld nodes achieved GFLOPS per node: %lf.\n", nodes, gflops / double(nodes));

  // We turn off validation temporarily because of the fills that aren't distribution aware.
  // tacoValidate<valType>(ctx, runtime, A, valType(n));
}

TACO_MAIN(valType)
