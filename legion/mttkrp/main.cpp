#include "legion.h"
#include "taco_mapper.h"
#include "legion_utils.h"

using namespace Legion;

typedef double valType;

// Defined by the generated TACO code.
void registerTacoTasks();

LogicalPartition placeLegionB(Context ctx, Runtime* runtime, LogicalRegion B, int32_t gridX, int32_t gridY, int32_t gridZ);
void computeLegion(Context ctx, Runtime* runtime, LogicalRegion A, LogicalRegion B, LogicalRegion C, LogicalRegion D, LogicalPartition BPartition);

void top_level_task(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  auto args = runtime->get_input_args();
  int n = -1;
  int gx = -1;
  int gy = -1;
  int gz = -1;

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
    if (strcmp(args.argv[i], "-gz") == 0) {
      gz = atoi(args.argv[++i]);
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
    std::cout << "Please provide a gris y size with -gy." << std::endl;
    return;
  }
  if (gz == -1) {
    std::cout << "Please provide a gris y size with -gy." << std::endl;
    return;
  }

  auto fspace = runtime->create_field_space(ctx);
  allocate_tensor_fields<valType>(ctx, runtime, fspace);
  auto aISpace = runtime->create_index_space(ctx, Rect<2>({0, 0}, {n - 1, n - 1}));
  auto bISpace = runtime->create_index_space(ctx, Rect<3>({0, 0, 0}, {n - 1, n - 1, n - 1}));
  auto cISpace = runtime->create_index_space(ctx, Rect<2>({0, 0}, {n - 1, n - 1}));
  auto dISpace = runtime->create_index_space(ctx, Rect<2>({0, 0}, {n - 1, n - 1}));
  auto A = runtime->create_logical_region(ctx, aISpace, fspace); runtime->attach_name(A, "A");
  auto B = runtime->create_logical_region(ctx, bISpace, fspace); runtime->attach_name(B, "B");
  auto C = runtime->create_logical_region(ctx, cISpace, fspace); runtime->attach_name(C, "C");
  auto D = runtime->create_logical_region(ctx, dISpace, fspace); runtime->attach_name(D, "D");

  tacoFill<valType>(ctx, runtime, B, 1);
  tacoFill<valType>(ctx, runtime, C, 1);
  tacoFill<valType>(ctx, runtime, D, 1);

  for (int i = 0; i < 10; i++) {
    tacoFill<valType>(ctx, runtime, A, 0);
    auto part = placeLegionB(ctx, runtime, B, gx, gy, gz);
    benchmark(ctx, runtime, [&]() { computeLegion(ctx, runtime, A, B, C, D, part); });
  }

  tacoValidate<valType>(ctx, runtime, A, valType(n * n));
}

TACO_MAIN(valType)
