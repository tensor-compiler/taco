#include "legion.h"
#include "taco_mapper.h"
#include "legion_utils.h"

using namespace Legion;

// Defined by the generated TACO code.
void registerTacoTasks();
LogicalPartition placeLegion(Context ctx, Runtime* runtime, LogicalRegion a);

void top_level_task(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  int n = 1024;
  auto fspace = runtime->create_field_space(ctx);
  allocate_tensor_fields<int32_t>(ctx, runtime, fspace);
  auto ispace = runtime->create_index_space(ctx, Rect<2>({0, 0}, {n - 1, n - 1}));
  auto A = runtime->create_logical_region(ctx, ispace, fspace); runtime->attach_name(A, "A");
  tacoFill(ctx, runtime, A, 0);
  auto part = placeLegion(ctx, runtime, A);
}

TACO_MAIN()