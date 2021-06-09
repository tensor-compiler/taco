#include "cublas_v2.h"
#include "cudalibs.h"
#include "taco_legion_header.h"
#include "taco_mapper.h"
#define TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))
using namespace Legion;
typedef FieldAccessor<READ_ONLY,int32_t,1,coord_t,Realm::AffineAccessor<int32_t,1,coord_t>> AccessorROint32_t1;
typedef FieldAccessor<READ_WRITE,int32_t,1,coord_t,Realm::AffineAccessor<int32_t,1,coord_t>> AccessorRWint32_t1;

struct task_1Args {
  int32_t b1_dimension;
};

__global__
void task_1DeviceKernel0(AccessorRWint32_t1 a_vals, AccessorROint32_t1 b_vals, int32_t b1_dimension, int32_t in) {

  int32_t bvar = blockIdx.x;
  int32_t tvar = (threadIdx.x % (32));
  int32_t wvar = (threadIdx.x / 32);
  if (threadIdx.x >= 256) {
    return;
  }

  for (int32_t f3 = 0; f3 < 8; f3++) {
    int32_t f2 = tvar * 8 + f3;
    int32_t f1 = wvar * 256 + f2;
    int32_t il = bvar * 2048 + f1;
    int32_t i = in * ((b1_dimension + 3) / 4) + il;
    Point<1> a_access_point = Point<1>(i);
    Point<1> b_access_point = Point<1>(i);
    if (i >= b1_dimension)
      break;

    if (i >= (in + 1) * ((b1_dimension + 3) / 4))
      break;

    a_vals[a_access_point] = b_vals[b_access_point];
  }
}

void task_1(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion a = regions[0];
  PhysicalRegion b = regions[1];

  int32_t in = task->index_point[0];
  task_1Args* args = (task_1Args*)(task->args);
  int32_t b1_dimension = args->b1_dimension;

  AccessorROint32_t1 b_vals(b, FID_VAL);
  AccessorRWint32_t1 a_vals(a, FID_VAL);

  task_1DeviceKernel0<<<(((b1_dimension + 3) / 4 + 2047) / 2048), (32 * 8)>>>(a_vals, b_vals, b1_dimension, in);
}

void computeLegion(Context ctx, Runtime* runtime, LogicalRegion a, LogicalRegion b) {
  auto a_index_space = get_index_space(a);
  int b1_dimension = runtime->get_index_space_domain(get_index_space(b)).hi()[0] + 1;
  auto b_index_space = get_index_space(b);

  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>(3);
  auto inIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(inIndexSpace));
  auto aDomain = runtime->get_index_space_domain(ctx, a_index_space);
  auto bDomain = runtime->get_index_space_domain(ctx, b_index_space);
  DomainPointColoring aColoring = DomainPointColoring();
  DomainPointColoring bColoring = DomainPointColoring();
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    int32_t in = (*itr)[0];
    Point<1> aStart = Point<1>((in * ((b1_dimension + 3) / 4)));
    Point<1> aEnd = Point<1>(TACO_MIN((in * ((b1_dimension + 3) / 4) + ((b1_dimension + 3) / 4 - 1)),aDomain.hi()[0]));
    Rect<1> aRect = Rect<1>(aStart, aEnd);
    if (!aDomain.contains(aRect.lo) || !aDomain.contains(aRect.hi)) aRect = aRect.make_empty();

    aColoring[(*itr)] = aRect;
    Point<1> bStart = Point<1>((in * ((b1_dimension + 3) / 4)));
    Point<1> bEnd = Point<1>(TACO_MIN((in * ((b1_dimension + 3) / 4) + ((b1_dimension + 3) / 4 - 1)),bDomain.hi()[0]));
    Rect<1> bRect = Rect<1>(bStart, bEnd);
    if (!bDomain.contains(bRect.lo) || !bDomain.contains(bRect.hi)) bRect = bRect.make_empty();

    bColoring[(*itr)] = bRect;
  }
  auto aPartition = runtime->create_index_partition(ctx, a_index_space, domain, aColoring, LEGION_DISJOINT_KIND);
  auto bPartition = runtime->create_index_partition(ctx, b_index_space, domain, bColoring, LEGION_DISJOINT_KIND);
  LogicalPartition aLogicalPartition = runtime->get_logical_partition(ctx, get_logical_region(a), aPartition);
  RegionRequirement aReq = RegionRequirement(aLogicalPartition, 0, READ_WRITE, EXCLUSIVE, get_logical_region(a));
  aReq.add_field(FID_VAL);
  LogicalPartition bLogicalPartition = runtime->get_logical_partition(ctx, get_logical_region(b), bPartition);
  RegionRequirement bReq = RegionRequirement(bLogicalPartition, 0, READ_ONLY, EXCLUSIVE, get_logical_region(b));
  bReq.add_field(FID_VAL);
  task_1Args taskArgsRaw;
  taskArgsRaw.b1_dimension = b1_dimension;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_1Args));
  IndexLauncher launcher = IndexLauncher(taskID(1), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(aReq);
  launcher.add_region_requirement(bReq);
  auto fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();

}
void registerTacoTasks() {
  {
    TaskVariantRegistrar registrar(taskID(1), "task_1");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<task_1>(registrar, "task_1");
  }
}
