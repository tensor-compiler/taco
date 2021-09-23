#include "legion.h"
#include "taco_mapper.h"
#include "legion_utils.h"

using namespace Legion;

typedef double valType;

// Defined by the generated TACO code.
void registerTacoTasks();

std::vector<LogicalPartition> partitionForplaceLegionA(Context ctx, Runtime* runtime, LogicalRegion A, int32_t rpoc);
void placeLegionA(Context ctx, Runtime* runtime, LogicalRegion A, LogicalPartition APartition, int32_t rpoc, int32_t c);

std::vector<LogicalPartition> partitionForplaceLegionB(Context ctx, Runtime* runtime, LogicalRegion B, int32_t rpoc);
void placeLegionB(Context ctx, Runtime* runtime, LogicalRegion B, LogicalPartition BPartition, int32_t rpoc, int32_t c);

std::vector<LogicalPartition> partitionForplaceLegionC(Context ctx, Runtime* runtime, LogicalRegion C, int32_t rpoc);
void placeLegionC(Context ctx, Runtime* runtime, LogicalRegion C, LogicalPartition CPartition, int32_t rpoc, int32_t c);

std::vector<LogicalPartition> partitionForcomputeLegion(Context ctx, Runtime* runtime, LogicalRegion A, LogicalRegion B, LogicalRegion C, int32_t rpoc, int32_t c);
void computeLegion(Context ctx, Runtime* runtime, LogicalRegion A, LogicalRegion B, LogicalRegion C, LogicalPartition APartition, LogicalPartition BPartition, LogicalPartition CPartition, int32_t rpoc, int32_t c, int32_t rpoc3);

void top_level_task(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  // Create the regions.
  auto args = runtime->get_input_args();
  int n = -1;
  int rpoc = -1;
  int rpoc3 = -1;
  int c = -1;
  // Parse input args.
  for (int i = 1; i < args.argc; i++) {
    if (strcmp(args.argv[i], "-n") == 0) {
      n = atoi(args.argv[++i]);
      continue;
    }
    if (strcmp(args.argv[i], "-rpoc") == 0) {
      rpoc = atoi(args.argv[++i]);
      continue;
    }
    if (strcmp(args.argv[i], "-rpoc3") == 0) {
      rpoc3 = atoi(args.argv[++i]);
      continue;
    }
    if (strcmp(args.argv[i], "-c") == 0) {
      c = atoi(args.argv[++i]);
      continue;
    }
  }
  // TODO (rohany): Improve these messages.
  if (n == -1) {
    std::cout << "Please provide an input matrix size with -n." << std::endl;
    return;
  }
  if (rpoc == -1) {
    std::cout << "Please provide a rpoc." << std::endl;
    return;
  }
  if (rpoc3 == -1) {
    std::cout << "Please provide a rpoc3." << std::endl;
    return;
  }
  if (c == -1) {
    std::cout << "Please provide a c." << std::endl;
    return;
  }

  initCuBLAS(ctx, runtime);

  auto fspace = runtime->create_field_space(ctx);
  allocate_tensor_fields<valType>(ctx, runtime, fspace);
  auto ispace = runtime->create_index_space(ctx, Rect<2>({0, 0}, {n - 1, n - 1}));
  auto A = runtime->create_logical_region(ctx, ispace, fspace); runtime->attach_name(A, "A");
  auto B = runtime->create_logical_region(ctx, ispace, fspace); runtime->attach_name(B, "B");
  auto C = runtime->create_logical_region(ctx, ispace, fspace); runtime->attach_name(C, "C");

  // Partition all tensors.
  auto aPart = partitionForplaceLegionA(ctx, runtime, A, rpoc)[0];
  auto bPart = partitionForplaceLegionB(ctx, runtime, B, rpoc)[0];
  auto cPart = partitionForplaceLegionC(ctx, runtime, C, rpoc)[0];

  auto parts = partitionForcomputeLegion(ctx, runtime, A, B, C, rpoc, c);

  std::vector<size_t> times;
  // Run the benchmark several times.
  for (int i = 0; i < 11; i++) {
    // TODO (rohany): We could potentially eliminate these fills to place the data right where
    //  we need it just like Johnson's algorithm. This would allow us to use larger values of c
    //  as well which might improve performance.
    tacoFill<valType>(ctx, runtime, A, aPart, 0);
    tacoFill<valType>(ctx, runtime, B, bPart, 1);
    tacoFill<valType>(ctx, runtime, C, cPart, 1);

    // Place the tensors.
    placeLegionA(ctx, runtime, A, aPart, rpoc, c);
    placeLegionB(ctx, runtime, B, bPart, rpoc, c);
    placeLegionC(ctx, runtime, C, cPart, rpoc, c);

    auto bench = [&]() {
      computeLegion(ctx, runtime, A, B, C, parts[0], parts[1], parts[2], rpoc, c, rpoc3);
      placeLegionA(ctx, runtime, A, aPart, rpoc, c);
    };

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

  tacoValidate<valType>(ctx, runtime, A, aPart, valType(n));
}

TACO_MAIN(valType)
