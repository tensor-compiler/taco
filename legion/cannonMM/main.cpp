#include "legion.h"
#include "taco_mapper.h"
#include "legion_utils.h"

using namespace Legion;

// Defined by the generated TACO code.
void registerTacoTasks();
LogicalPartition placeLegionA(Context ctx, Runtime* runtime, LogicalRegion a);
LogicalPartition placeLegionB(Context ctx, Runtime* runtime, LogicalRegion b);
LogicalPartition placeLegionC(Context ctx, Runtime* runtime, LogicalRegion c);
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
    // TODO (rohany): Add a flag to do the validation or not.
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
  tacoFill(ctx, runtime, A, 0); tacoFill(ctx, runtime, B, 1); tacoFill(ctx, runtime, C, 1);

  // Place the tensors.
  auto part = placeLegionA(ctx, runtime, A);
  placeLegionA(ctx, runtime, B);
  placeLegionA(ctx, runtime, C);

  // Compute on the tensors.
  benchmark([&]() { computeLegion(ctx, runtime, A, B, C, part); });

  auto a_reg = getRegionToWrite(ctx, runtime, A, A);
  FieldAccessor<READ_WRITE,int32_t,2,coord_t, Realm::AffineAccessor<int32_t, 2, coord_t>> a_rw(a_reg, FID_VAL);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      assert(a_rw[Point<2>(i, j)] == n);
    }
  }
  runtime->unmap_region(ctx, a_reg);
}

TACO_MAIN()
