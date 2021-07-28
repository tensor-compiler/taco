#include "legion.h"
#include "taco_mapper.h"
#include "legion_utils.h"

using namespace Legion;

typedef double valType;

// Defined by the generated TACO code.
void registerTacoTasks();
LogicalPartition partitionLegion(Context ctx, Runtime* runtime, LogicalRegion A, int32_t rpoc);
LogicalPartition placeLegionA(Context ctx, Runtime* runtime, LogicalRegion A, int32_t rpoc, int32_t c);
LogicalPartition placeLegionB(Context ctx, Runtime* runtime, LogicalRegion B, int32_t rpoc, int32_t c);
LogicalPartition placeLegionC(Context ctx, Runtime* runtime, LogicalRegion C, int32_t rpoc, int32_t c);
void computeLegion(Context ctx, Runtime* runtime, LogicalRegion A, LogicalRegion B, LogicalRegion C, int32_t rpoc, int32_t c, int32_t rpoc3);

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
  auto aPart = partitionLegion(ctx, runtime, A, rpoc);
  auto bPart = partitionLegion(ctx, runtime, B, rpoc);
  auto cPart = partitionLegion(ctx, runtime, C, rpoc);

  std::vector<size_t> times;
  // Run the benchmark several times.
  for (int i = 0; i < 10; i++) {
    tacoFill<valType>(ctx, runtime, A, aPart, 0);
    tacoFill<valType>(ctx, runtime, B, bPart, 1);
    tacoFill<valType>(ctx, runtime, C, cPart, 1);

    // Place the tensors.
    placeLegionA(ctx, runtime, A, rpoc, c);
    placeLegionB(ctx, runtime, B, rpoc, c);
    placeLegionC(ctx, runtime, C, rpoc, c);

    benchmark(ctx, runtime, times, [&]() {
      computeLegion(ctx, runtime, A, B, C, rpoc, c, rpoc3);
      placeLegionA(ctx, runtime, A, rpoc, c);
    });
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
