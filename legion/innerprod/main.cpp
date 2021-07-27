#include "legion.h"
#include "taco_mapper.h"
#include "legion_utils.h"

using namespace Legion;

typedef double valType;

void registerTacoTasks();
LogicalPartition partition3Tensor(Context ctx, Runtime* runtime, LogicalRegion T, int32_t pieces);
valType computeLegion(Context ctx, Runtime* runtime, LogicalRegion B, LogicalRegion C, LogicalPartition bPartition, LogicalPartition cPartition);

void top_level_task(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  // Create the regions.
  auto args = runtime->get_input_args();
  int n = -1;
  int pieces = -1;

  // Parse input args.
  for (int i = 1; i < args.argc; i++) {
    if (strcmp(args.argv[i], "-n") == 0) {
      n = atoi(args.argv[++i]);
      continue;
    }
    if (strcmp(args.argv[i], "-pieces") == 0) {
      pieces = atoi(args.argv[++i]);
      continue;
    }
  }
  if (n == -1) {
    std::cout << "Please provide an input matrix size with -n." << std::endl;
    return;
  }
  if (pieces == -1) {
    std::cout << "Please provide the number of pieces with -pieces." << std::endl;
    return;
  }

  auto fspace = runtime->create_field_space(ctx);
  allocate_tensor_fields<valType>(ctx, runtime, fspace);

  auto ispace = runtime->create_index_space(ctx, Rect<3>({0, 0, 0}, {n - 1, n - 1, n - 1}));
  auto B = runtime->create_logical_region(ctx, ispace, fspace); runtime->attach_name(B, "B");
  auto C = runtime->create_logical_region(ctx, ispace, fspace); runtime->attach_name(C, "C");

  auto bPart = partition3Tensor(ctx, runtime, B, pieces);
  auto cPart = partition3Tensor(ctx, runtime, C, pieces);

  tacoFill<valType>(ctx, runtime, B, bPart, valType(1));
  tacoFill<valType>(ctx, runtime, C, cPart, valType(1));

  // Run one iteration to warm up the system.
  computeLegion(ctx, runtime, B, C, bPart, cPart);

  std::vector<size_t> times;
  for (int i = 0; i < 10; i++) {
    benchmark(ctx, runtime, times, [&]() {
      auto res = computeLegion(ctx, runtime, B, C, bPart, cPart);
      assert(res == valType(valType(n) * valType(n) * valType(n)));
    });
  }
  // benchmarkAsyncCall(ctx, runtime, times, [&]() {
  //   for (int i = 0; i < 10; i++) {
  //     auto res = computeLegion(ctx, runtime, B, C, bPart, cPart);
  //     assert(res == valType(valType(n) * valType(n) * valType(n)));
  //   }
  // });

  size_t elems = [](size_t n) { return 2 * n * n * n; }(n);
  size_t bytes = elems * sizeof(valType);
  double gbytes = double(bytes) / 1e9;
  // auto avgTimeS = (double(times[0]) / 10.f) / 1e3;
  auto avgTimeS = (double(average(times))) / 1e3;
  double bw = gbytes / (avgTimeS);
  auto nodes = runtime->select_tunable_value(ctx, Mapping::DefaultMapper::DEFAULT_TUNABLE_NODE_COUNT).get<size_t>();
  LEGION_PRINT_ONCE(runtime, ctx, stdout, "On %ld nodes achieved GB/s BW per node: %lf.\n", nodes, bw / double(nodes));
}

TACO_MAIN(valType)
