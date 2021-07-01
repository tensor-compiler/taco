#include "legion.h"
#include "taco_mapper.h"
#include "legion_utils.h"

using namespace Legion;

typedef double valType;

// Defined by the generated TACO code.
void registerTacoTasks();
LogicalPartition partitionLegion(Context ctx, Runtime* runtime, LogicalRegion a, int32_t gridDim);
LogicalPartition placeLegionA(Context ctx, Runtime* runtime, LogicalRegion a, int32_t gdim);
LogicalPartition placeLegionB(Context ctx, Runtime* runtime, LogicalRegion b, int32_t gdim);
LogicalPartition placeLegionC(Context ctx, Runtime* runtime, LogicalRegion c, int32_t gdim);
void computeLegion(Context ctx, Runtime* runtime, LogicalRegion a, LogicalRegion b, LogicalRegion c, int gdim);

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

  auto fspace = runtime->create_field_space(ctx);
  allocate_tensor_fields<valType>(ctx, runtime, fspace);
  auto ispace = runtime->create_index_space(ctx, Rect<2>({0, 0}, {n - 1, n - 1}));
  auto A = runtime->create_logical_region(ctx, ispace, fspace); runtime->attach_name(A, "A");
  auto B = runtime->create_logical_region(ctx, ispace, fspace); runtime->attach_name(B, "B");
  auto C = runtime->create_logical_region(ctx, ispace, fspace); runtime->attach_name(C, "C");

  // Partition all of the tensors.
  auto aPart = partitionLegion(ctx, runtime, A, gdim);
  auto bPart = partitionLegion(ctx, runtime, B, gdim);
  auto cPart = partitionLegion(ctx, runtime, C, gdim);

  std::vector<size_t> times;
  for (int i = 0; i < 10; i++) {
    tacoFill<valType>(ctx, runtime, A, aPart, 0);
    tacoFill<valType>(ctx, runtime, B, bPart, 1);
    tacoFill<valType>(ctx, runtime, C, cPart, 1);

    // Place the tensors.
    placeLegionA(ctx, runtime, A, gdim);
    placeLegionB(ctx, runtime, B, gdim);
    placeLegionC(ctx, runtime, C, gdim);

    // Compute on the tensors.
    benchmark(ctx, runtime, times, [&]() {
      computeLegion(ctx, runtime, A, B, C, gdim);
      // Call the placement function again to force reduction along each slice of A.
      placeLegionA(ctx, runtime, A, gdim);
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
