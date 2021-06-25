#include "legion.h"
#include "taco_mapper.h"
#include "legion_utils.h"

using namespace Legion;

typedef double valType;

// Defined by the generated TACO code.
void registerTacoTasks();

LogicalPartition placeLegionA(Context ctx, Runtime* runtime, LogicalRegion A, int32_t pieces);
void computeLegion(Context ctx, Runtime* runtime, LogicalRegion A, LogicalRegion B, LogicalRegion C, LogicalPartition aPartition);

void top_level_task(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
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

  auto aISpace = runtime->create_index_space(ctx, Rect<3>({0, 0, 0}, {n - 1, n - 1, n - 1}));
  auto bISpace = runtime->create_index_space(ctx, Rect<3>({0, 0, 0}, {n - 1, n - 1, n - 1}));
  auto cISpace = runtime->create_index_space(ctx, Rect<2>({0, 0}, {n - 1, n - 1}));
  auto A = runtime->create_logical_region(ctx, aISpace, fspace); runtime->attach_name(A, "A");
  auto B = runtime->create_logical_region(ctx, bISpace, fspace); runtime->attach_name(B, "B");
  auto C = runtime->create_logical_region(ctx, cISpace, fspace); runtime->attach_name(C, "C");

  tacoFill<valType>(ctx, runtime, B, 1);
  tacoFill<valType>(ctx, runtime, C, 1);

  for (int i = 0; i < 10; i++) {
    tacoFill<valType>(ctx, runtime, A, 0);
    auto part = placeLegionA(ctx, runtime, A, pieces);
    benchmark(ctx, runtime, [&]() { computeLegion(ctx, runtime, A, B, C, part); });
  }

  tacoValidate<valType>(ctx, runtime, A, valType(n));
}

TACO_MAIN(valType)
