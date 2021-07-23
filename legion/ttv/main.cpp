#include "legion.h"
#include "taco_mapper.h"
#include "legion_utils.h"

using namespace Legion;

typedef double valType;

// Defined by the generated TACO code.
void registerTacoTasks();
LogicalPartition partitionLegionA(Context ctx, Runtime* runtime, LogicalRegion A, int32_t gridX, int32_t gridY);
LogicalPartition partitionLegionB(Context ctx, Runtime* runtime, LogicalRegion B, int32_t gridX, int32_t gridY);
LogicalPartition placeLegionA(Context ctx, Runtime* runtime, LogicalRegion A, int32_t gridX, int32_t gridY);
LogicalPartition placeLegionB(Context ctx, Runtime* runtime, LogicalRegion B, int32_t gridX, int32_t gridY);
LogicalPartition placeLegionC(Context ctx, Runtime* runtime, LogicalRegion C, int32_t gridX, int32_t gridY);
void computeLegion(Context ctx, Runtime* runtime, LogicalRegion A, LogicalRegion B, LogicalRegion C, LogicalPartition bPartition, LogicalPartition aPartition, LogicalPartition cPartition);

void top_level_task(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  // Create the regions.
  auto args = runtime->get_input_args();
  int n = -1;
  int gx = -1;
  int gy = -1;
  bool validate = false;
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
    if (strcmp(args.argv[i], "-validate") == 0) {
      validate = true;
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
    std::cout << "Please provide a grid y size with -gy." << std::endl;
    return;
  }

  auto fspace = runtime->create_field_space(ctx);
  allocate_tensor_fields<valType>(ctx, runtime, fspace);

  auto aISpace = runtime->create_index_space(ctx, Rect<2>({0, 0}, {n - 1, n - 1}));
  auto bISpace = runtime->create_index_space(ctx, Rect<3>({0, 0, 0}, {n - 1, n - 1, n - 1}));
  // Create a "replicated" version of C.
  auto cISpace = runtime->create_index_space(ctx, Rect<1>({0, (n * gx * gy) - 1}));
  auto A = runtime->create_logical_region(ctx, aISpace, fspace); runtime->attach_name(A, "A");
  auto B = runtime->create_logical_region(ctx, bISpace, fspace); runtime->attach_name(B, "B");
  auto C = runtime->create_logical_region(ctx, cISpace, fspace); runtime->attach_name(C, "C");

  // Create initial partitions.
  auto aPart = partitionLegionA(ctx, runtime, A, gx, gy);
  auto bPart = partitionLegionB(ctx, runtime, B, gx, gy);
  // Create a "replicated" partition of C.
  DomainPointColoring cColoring;
  Point<2> lowerBound = Point<2>(0, 0);
  Point<2> upperBound = Point<2>((gx - 1), (gy - 1));
  DomainT<2> dom(Rect<2>{lowerBound, upperBound});
  for (PointInDomainIterator<2> itr(dom); itr(); itr++) {
    int32_t in = (*itr)[0];
    int32_t jn = (*itr)[1];
    int idx = in * gy + jn;
    auto start = idx * n;
    auto end = (idx + 1) * n - 1;
    cColoring[*itr] = Rect<1>(start, end);
  }
  auto cIndexPart = runtime->create_index_partition(ctx, cISpace, dom, cColoring, LEGION_DISJOINT_COMPLETE_KIND);
  auto cPart = runtime->get_logical_partition(ctx, C, cIndexPart);

  tacoFill<valType>(ctx, runtime, A, aPart, 0);
  tacoFill<valType>(ctx, runtime, B, bPart, 1);
  tacoFill<valType>(ctx, runtime, C, cPart, 1);

  placeLegionA(ctx, runtime, A, gx, gy);
  placeLegionB(ctx, runtime, B, gx, gy);

  // Run the function once to warm up the runtime system and validate the result.
  computeLegion(ctx, runtime, A, B, C, bPart, aPart, cPart);
  tacoValidate<valType>(ctx, runtime, A, aPart, valType(n));

  std::vector<size_t> times;
  benchmarkAsyncCall(ctx, runtime, times, [&]() {
    for (int i = 0; i < 10; i++) {
      computeLegion(ctx, runtime, A, B, C, bPart, aPart, cPart);
    }
  });

  // Calculate the total bandwidth.
  size_t elems = [](size_t n) { return n * n + n * n * n + n; }(n);
  size_t bytes = elems * sizeof(valType);
  double gbytes = double(bytes) / 1e9;
  auto avgTimeS = (double(times[0]) / 10.f) / 1e3;
  double bw = gbytes / (avgTimeS);
  auto nodes = runtime->select_tunable_value(ctx, Mapping::DefaultMapper::DEFAULT_TUNABLE_NODE_COUNT).get<size_t>();
  LEGION_PRINT_ONCE(runtime, ctx, stdout, "On %ld nodes achieved GB/s BW per node: %lf.\n", nodes, bw / double(nodes));
}

TACO_MAIN(valType)
