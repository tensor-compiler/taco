#include "legion.h"
#include "taco_mapper.h"
#include "legion_utils.h"

using namespace Legion;

typedef double valType;

// Defined by the generated TACO code.
void registerTacoTasks();
LogicalPartition placeLegionA(Context ctx, Runtime* runtime, LogicalRegion a, int gx, int gy);
LogicalPartition placeLegionB(Context ctx, Runtime* runtime, LogicalRegion b, int gx, int gy);
LogicalPartition placeLegionC(Context ctx, Runtime* runtime, LogicalRegion c, int gx, int gy);
void computeLegion(Context ctx, Runtime* runtime, LogicalRegion a, LogicalRegion b, LogicalRegion c, LogicalPartition aPartition);

void top_level_task(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  // Create the regions.
  auto args = runtime->get_input_args();
  int n = -1;
  int gx = -1;
  int gy = -1;
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
    // TODO (rohany): Add a flag to do the validation or not.
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
    std::cout << "Please provide a gris y size with -gy." << std::endl;
    return;
  }

  auto fspace = runtime->create_field_space(ctx);
  allocate_tensor_fields<valType>(ctx, runtime, fspace);
  auto ispace = runtime->create_index_space(ctx, Rect<2>({0, 0}, {n - 1, n - 1}));
  auto A = runtime->create_logical_region(ctx, ispace, fspace); runtime->attach_name(A, "A");
  auto B = runtime->create_logical_region(ctx, ispace, fspace); runtime->attach_name(B, "B");
  auto C = runtime->create_logical_region(ctx, ispace, fspace); runtime->attach_name(C, "C");
  tacoFill<valType>(ctx, runtime, A, 0); tacoFill<valType>(ctx, runtime, B, 1); tacoFill<valType>(ctx, runtime, C, 1);

  // Place the tensors.
  auto part = placeLegionA(ctx, runtime, A, gx, gy);
  placeLegionB(ctx, runtime, B, gx, gy);
  placeLegionC(ctx, runtime, C, gx, gy);

  // Compute on the tensors.
  benchmark(ctx, runtime, [&]() { computeLegion(ctx, runtime, A, B, C, part); });

  tacoValidate<valType>(ctx, runtime, A, valType(n));
}

TACO_MAIN(valType)