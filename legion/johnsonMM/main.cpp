#include "legion.h"
#include "taco_mapper.h"
#include "legion_utils.h"

using namespace Legion;

typedef double valType;

// Defined by the generated TACO code.
void registerTacoTasks();
LogicalPartition placeLegionA(Context ctx, Runtime* runtime, LogicalRegion a, int gdim);
LogicalPartition placeLegionB(Context ctx, Runtime* runtime, LogicalRegion b, int gdim);
LogicalPartition placeLegionC(Context ctx, Runtime* runtime, LogicalRegion c, int gdim);
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
  tacoFill<valType>(ctx, runtime, A, 0); tacoFill<valType>(ctx, runtime, B, 1); tacoFill<valType>(ctx, runtime, C, 1);

  // Place the tensors.
  placeLegionA(ctx, runtime, A, gdim); placeLegionB(ctx, runtime, B, gdim); placeLegionC(ctx, runtime, C, gdim);

  // Compute on the tensors.
  benchmark([&]() { computeLegion(ctx, runtime, A, B, C, gdim); });

  tacoValidate<valType>(ctx, runtime, A, valType(n));
}

TACO_MAIN(valType)
