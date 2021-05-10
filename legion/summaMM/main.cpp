#include "legion.h"
#include "taco_mapper.h"
#include "legion_utils.h"

using namespace Legion;

// Defined by the generated TACO code.
void registerTacoTasks();
void computeLegion(Context ctx, Runtime* runtime, LogicalRegion a, LogicalRegion b, LogicalRegion c, LogicalPartition aPartition);

void top_level_task(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  // Create the regions.
  auto args = runtime->get_input_args();
  int n = -1;
  // Parse input args.
  for (int i = 1; i < args.argc; i++) {
    if (strcmp(args.argv[i], "-n") == 0) {
      n = atoi(args.argv[++i]);
      continue;
    }
  }
  if (n == -1) {
    std::cout << "Please provide an input matrix size with -n." << std::endl;
    return;
  }

  auto fspace = runtime->create_field_space(ctx);
  allocate_tensor_fields<int32_t>(ctx, runtime, fspace);
  auto ispace = runtime->create_index_space(ctx, Rect<2>({0, 0}, {n - 1, n - 1}));
  auto A = runtime->create_logical_region(ctx, ispace, fspace); runtime->attach_name(A, "A");
  auto B = runtime->create_logical_region(ctx, ispace, fspace); runtime->attach_name(B, "B");
  auto C = runtime->create_logical_region(ctx, ispace, fspace); runtime->attach_name(C, "C");
  runtime->fill_field(ctx, A, A, FID_VAL, 0);
  runtime->fill_field(ctx, B, B, FID_VAL, 1);
  runtime->fill_field(ctx, C, C, FID_VAL, 1);

  // TODO (rohany): Include placement code here.
  // For now, just use an equal partition of A.
  auto icspace = runtime->create_index_space(ctx, Rect<2>(Point<2>(0, 0), Point<2>(1, 1)));
  auto part = runtime->create_equal_partition(ctx, A.get_index_space(), icspace);
  computeLegion(ctx, runtime, A, B, C, runtime->get_logical_partition(A, part));

  // TODO (rohany): Include validation code here.
}

TACO_MAIN()