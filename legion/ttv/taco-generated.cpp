#include "taco_legion_header.h"
#include "taco_mapper.h"
#define TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))
using namespace Legion;
typedef FieldAccessor<READ_ONLY,double,1,coord_t,Realm::AffineAccessor<double,1,coord_t>> AccessorROdouble1;
typedef FieldAccessor<READ_WRITE,double,2,coord_t,Realm::AffineAccessor<double,2,coord_t>> AccessorRWdouble2;
typedef FieldAccessor<READ_ONLY,double,3,coord_t,Realm::AffineAccessor<double,3,coord_t>> AccessorROdouble3;
typedef FieldAccessor<READ_WRITE,double,2,coord_t,Realm::AffineAccessor<double,2,coord_t>> AccessorRWdouble2;

struct task_1Args {
  int32_t gridX;
  int32_t gridY;
};
struct task_2Args {
  int32_t gridX;
  int32_t gridY;
};
struct task_3Args {
  int32_t gridX;
  int32_t gridY;
};
struct task_4Args {
  int32_t A1_dimension;
  int32_t A2_dimension;
  int32_t B1_dimension;
  int32_t B2_dimension;
  int32_t B3_dimension;
  int32_t C1_dimension;
  int32_t gridX;
  int32_t gridY;
};

LogicalPartition partitionLegionA(Context ctx, Runtime* runtime, LogicalRegion A, int32_t gridX, int32_t gridY) {
  int A1_dimension = runtime->get_index_space_domain(get_index_space(A)).hi()[0] + 1;
  int A2_dimension = runtime->get_index_space_domain(get_index_space(A)).hi()[1] + 1;
  auto A_index_space = get_index_space(A);

  Point<2> lowerBound = Point<2>(0, 0);
  Point<2> upperBound = Point<2>((gridX - 1), (gridY - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<2>(lowerBound, upperBound));
  DomainT<2> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<2>(distFusedIndexSpace));
  auto ADomain = runtime->get_index_space_domain(ctx, A_index_space);
  DomainPointColoring AColoring = DomainPointColoring();
  for (PointInDomainIterator<2> itr = PointInDomainIterator<2>(domain); itr.valid(); itr++) {
    int32_t in = (*itr)[0];
    int32_t jn = (*itr)[1];
    Point<2> AStart = Point<2>((in * ((A1_dimension + (gridX - 1)) / gridX) + 0 / gridX), (jn * ((A2_dimension + (gridY - 1)) / gridY) + 0 / gridY));
    Point<2> AEnd = Point<2>(TACO_MIN((in * ((A1_dimension + (gridX - 1)) / gridX) + ((A1_dimension + (gridX - 1)) / gridX - 1)), ADomain.hi()[0]), TACO_MIN((jn * ((A2_dimension + (gridY - 1)) / gridY) + ((A2_dimension + (gridY - 1)) / gridY - 1)), ADomain.hi()[1]));
    Rect<2> ARect = Rect<2>(AStart, AEnd);
    if (!ADomain.contains(ARect.lo) || !ADomain.contains(ARect.hi)) {
      ARect = ARect.make_empty();
    }
    AColoring[(*itr)] = ARect;
  }
  auto APartition = runtime->create_index_partition(ctx, A_index_space, domain, AColoring, LEGION_DISJOINT_COMPLETE_KIND);
  return runtime->get_logical_partition(ctx, get_logical_region(A), APartition);
}

LogicalPartition partitionLegionB(Context ctx, Runtime* runtime, LogicalRegion B, int32_t gridX, int32_t gridY) {
  int B1_dimension = runtime->get_index_space_domain(get_index_space(B)).hi()[0] + 1;
  int B2_dimension = runtime->get_index_space_domain(get_index_space(B)).hi()[1] + 1;
  int B3_dimension = runtime->get_index_space_domain(get_index_space(B)).hi()[2] + 1;
  auto B_index_space = get_index_space(B);

  Point<2> lowerBound = Point<2>(0, 0);
  Point<2> upperBound = Point<2>((gridX - 1), (gridY - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<2>(lowerBound, upperBound));
  DomainT<2> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<2>(distFusedIndexSpace));
  auto BDomain = runtime->get_index_space_domain(ctx, B_index_space);
  DomainPointColoring BColoring = DomainPointColoring();
  for (PointInDomainIterator<2> itr = PointInDomainIterator<2>(domain); itr.valid(); itr++) {
    int32_t in = (*itr)[0];
    int32_t jn = (*itr)[1];
    Point<3> BStart = Point<3>((in * ((B1_dimension + (gridX - 1)) / gridX) + 0 / gridX), (jn * ((B2_dimension + (gridY - 1)) / gridY) + 0 / gridY), 0);
    Point<3> BEnd = Point<3>(TACO_MIN((in * ((B1_dimension + (gridX - 1)) / gridX) + ((B1_dimension + (gridX - 1)) / gridX - 1)), BDomain.hi()[0]), TACO_MIN((jn * ((B2_dimension + (gridY - 1)) / gridY) + ((B2_dimension + (gridY - 1)) / gridY - 1)), BDomain.hi()[1]), TACO_MIN(B3_dimension, BDomain.hi()[2]));
    Rect<3> BRect = Rect<3>(BStart, BEnd);
    if (!BDomain.contains(BRect.lo) || !BDomain.contains(BRect.hi)) {
      BRect = BRect.make_empty();
    }
    BColoring[(*itr)] = BRect;
  }
  auto BPartition = runtime->create_index_partition(ctx, B_index_space, domain, BColoring, LEGION_DISJOINT_COMPLETE_KIND);
  return runtime->get_logical_partition(ctx, get_logical_region(B), BPartition);
}

void task_1(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion A = regions[0];

  int32_t distFused = task->index_point[0];
  task_1Args* args = (task_1Args*)(task->args);
  int32_t gridX = args->gridX;
  int32_t gridY = args->gridY;


  int32_t in = getIndexPoint(task, 0);
  int32_t jn = getIndexPoint(task, 1);
}

LogicalPartition placeLegionA(Context ctx, Runtime* runtime, LogicalRegion A, int32_t gridX, int32_t gridY) {
  int A1_dimension = runtime->get_index_space_domain(get_index_space(A)).hi()[0] + 1;
  int A2_dimension = runtime->get_index_space_domain(get_index_space(A)).hi()[1] + 1;
  auto A_index_space = get_index_space(A);

  Point<2> lowerBound = Point<2>(0, 0);
  Point<2> upperBound = Point<2>((gridX - 1), (gridY - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<2>(lowerBound, upperBound));
  DomainT<2> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<2>(distFusedIndexSpace));
  auto ADomain = runtime->get_index_space_domain(ctx, A_index_space);
  DomainPointColoring AColoring = DomainPointColoring();
  for (PointInDomainIterator<2> itr = PointInDomainIterator<2>(domain); itr.valid(); itr++) {
    int32_t in = (*itr)[0];
    int32_t jn = (*itr)[1];
    Point<2> AStart = Point<2>((in * ((A1_dimension + (gridX - 1)) / gridX) + 0 / gridX), (jn * ((A2_dimension + (gridY - 1)) / gridY) + 0 / gridY));
    Point<2> AEnd = Point<2>(TACO_MIN((in * ((A1_dimension + (gridX - 1)) / gridX) + ((A1_dimension + (gridX - 1)) / gridX - 1)), ADomain.hi()[0]), TACO_MIN((jn * ((A2_dimension + (gridY - 1)) / gridY) + ((A2_dimension + (gridY - 1)) / gridY - 1)), ADomain.hi()[1]));
    Rect<2> ARect = Rect<2>(AStart, AEnd);
    if (!ADomain.contains(ARect.lo) || !ADomain.contains(ARect.hi)) {
      ARect = ARect.make_empty();
    }
    AColoring[(*itr)] = ARect;
  }
  auto APartition = runtime->create_index_partition(ctx, A_index_space, domain, AColoring, LEGION_COMPUTE_KIND);
  LogicalPartition ALogicalPartition = runtime->get_logical_partition(ctx, get_logical_region(A), APartition);
  RegionRequirement AReq = RegionRequirement(ALogicalPartition, 0, READ_ONLY, EXCLUSIVE, get_logical_region(A));
  AReq.add_field(FID_VAL);
  task_1Args taskArgsRaw;
  taskArgsRaw.gridX = gridX;
  taskArgsRaw.gridY = gridY;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_1Args));
  IndexLauncher launcher = IndexLauncher(taskID(1), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(AReq);
  auto fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  return runtime->get_logical_partition(ctx, get_logical_region(A), APartition);

}

void task_2(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion B = regions[0];

  int32_t distFused = task->index_point[0];
  task_2Args* args = (task_2Args*)(task->args);
  int32_t gridX = args->gridX;
  int32_t gridY = args->gridY;


  int32_t in = getIndexPoint(task, 0);
  int32_t jn = getIndexPoint(task, 1);
}

LogicalPartition placeLegionB(Context ctx, Runtime* runtime, LogicalRegion B, int32_t gridX, int32_t gridY) {
  int B1_dimension = runtime->get_index_space_domain(get_index_space(B)).hi()[0] + 1;
  int B2_dimension = runtime->get_index_space_domain(get_index_space(B)).hi()[1] + 1;
  int B3_dimension = runtime->get_index_space_domain(get_index_space(B)).hi()[2] + 1;
  auto B_index_space = get_index_space(B);

  Point<2> lowerBound = Point<2>(0, 0);
  Point<2> upperBound = Point<2>((gridX - 1), (gridY - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<2>(lowerBound, upperBound));
  DomainT<2> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<2>(distFusedIndexSpace));
  auto BDomain = runtime->get_index_space_domain(ctx, B_index_space);
  DomainPointColoring BColoring = DomainPointColoring();
  for (PointInDomainIterator<2> itr = PointInDomainIterator<2>(domain); itr.valid(); itr++) {
    int32_t in = (*itr)[0];
    int32_t jn = (*itr)[1];
    Point<3> BStart = Point<3>((in * ((B1_dimension + (gridX - 1)) / gridX) + 0 / gridX), (jn * ((B2_dimension + (gridY - 1)) / gridY) + 0 / gridY), 0);
    Point<3> BEnd = Point<3>(TACO_MIN((in * ((B1_dimension + (gridX - 1)) / gridX) + ((B1_dimension + (gridX - 1)) / gridX - 1)), BDomain.hi()[0]), TACO_MIN((jn * ((B2_dimension + (gridY - 1)) / gridY) + ((B2_dimension + (gridY - 1)) / gridY - 1)), BDomain.hi()[1]), TACO_MIN(B3_dimension, BDomain.hi()[2]));
    Rect<3> BRect = Rect<3>(BStart, BEnd);
    if (!BDomain.contains(BRect.lo) || !BDomain.contains(BRect.hi)) {
      BRect = BRect.make_empty();
    }
    BColoring[(*itr)] = BRect;
  }
  auto BPartition = runtime->create_index_partition(ctx, B_index_space, domain, BColoring, LEGION_COMPUTE_KIND);
  LogicalPartition BLogicalPartition = runtime->get_logical_partition(ctx, get_logical_region(B), BPartition);
  RegionRequirement BReq = RegionRequirement(BLogicalPartition, 0, READ_ONLY, EXCLUSIVE, get_logical_region(B));
  BReq.add_field(FID_VAL);
  task_2Args taskArgsRaw;
  taskArgsRaw.gridX = gridX;
  taskArgsRaw.gridY = gridY;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_2Args));
  IndexLauncher launcher = IndexLauncher(taskID(2), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(BReq);
  auto fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  return runtime->get_logical_partition(ctx, get_logical_region(B), BPartition);

}

void task_3(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion C = regions[0];

  int32_t distFused = task->index_point[0];
  task_3Args* args = (task_3Args*)(task->args);
  int32_t gridX = args->gridX;
  int32_t gridY = args->gridY;


  int32_t jn = getIndexPoint(task, 0);
  int32_t kn = getIndexPoint(task, 1);
}

LogicalPartition placeLegionC(Context ctx, Runtime* runtime, LogicalRegion C, int32_t gridX, int32_t gridY) {
  int C1_dimension = runtime->get_index_space_domain(get_index_space(C)).hi()[0] + 1;
  auto C_index_space = get_index_space(C);

  Point<2> lowerBound = Point<2>(0, 0);
  Point<2> upperBound = Point<2>((gridX - 1), (gridY - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<2>(lowerBound, upperBound));
  DomainT<2> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<2>(distFusedIndexSpace));
  auto CDomain = runtime->get_index_space_domain(ctx, C_index_space);
  DomainPointColoring CColoring = DomainPointColoring();
  for (PointInDomainIterator<2> itr = PointInDomainIterator<2>(domain); itr.valid(); itr++) {
    Point<1> CStart = Point<1>(0);
    Point<1> CEnd = Point<1>(TACO_MIN(C1_dimension, CDomain.hi()[0]));
    Rect<1> CRect = Rect<1>(CStart, CEnd);
    if (!CDomain.contains(CRect.lo) || !CDomain.contains(CRect.hi)) {
      CRect = CRect.make_empty();
    }
    CColoring[(*itr)] = CRect;
  }
  auto CPartition = runtime->create_index_partition(ctx, C_index_space, domain, CColoring, LEGION_COMPUTE_KIND);
  LogicalPartition CLogicalPartition = runtime->get_logical_partition(ctx, get_logical_region(C), CPartition);
  RegionRequirement CReq = RegionRequirement(CLogicalPartition, 0, READ_ONLY, EXCLUSIVE, get_logical_region(C));
  CReq.add_field(FID_VAL);
  task_3Args taskArgsRaw;
  taskArgsRaw.gridX = gridX;
  taskArgsRaw.gridY = gridY;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_3Args));
  IndexLauncher launcher = IndexLauncher(taskID(3), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(CReq);
  auto fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  return runtime->get_logical_partition(ctx, get_logical_region(C), CPartition);

}

void task_4(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion A = regions[0];
  PhysicalRegion B = regions[1];
  PhysicalRegion C = regions[2];

  int32_t distFused = task->index_point[0];
  task_4Args* args = (task_4Args*)(task->args);
  int32_t A1_dimension = args->A1_dimension;
  int32_t A2_dimension = args->A2_dimension;
  int32_t B1_dimension = args->B1_dimension;
  int32_t B2_dimension = args->B2_dimension;
  int32_t B3_dimension = args->B3_dimension;
  int32_t C1_dimension = args->C1_dimension;
  int32_t gridX = args->gridX;
  int32_t gridY = args->gridY;

  AccessorROdouble1 C_vals(C, FID_VAL);
  AccessorROdouble3 B_vals(B, FID_VAL);
  AccessorRWdouble2 A_vals(A, FID_VAL);

  int32_t in = getIndexPoint(task, 0);
  int32_t jn = getIndexPoint(task, 1);
  #pragma omp parallel for schedule(runtime)
  for (int32_t ii = ((0 / gridX) / 4); ii < (((B1_dimension + (gridX - 1)) / gridX + 3) / 4); ii++) {
    int32_t il = ii * 4 + 0 / gridX;
    if ((in * ((B1_dimension + (gridX - 1)) / gridX) + il) + ((jn * ((B2_dimension + (gridY - 1)) / gridY) + (B2_dimension + (gridY - 1)) / gridY) - (jn * ((B2_dimension + (gridY - 1)) / gridY) + 0 / gridY)) >= B1_dimension || (jn * ((B2_dimension + (gridY - 1)) / gridY) + (B2_dimension + (gridY - 1)) / gridY) + ((jn * ((B2_dimension + (gridY - 1)) / gridY) + (B2_dimension + (gridY - 1)) / gridY) - (jn * ((B2_dimension + (gridY - 1)) / gridY) + 0 / gridY)) >= B2_dimension) {
      for (int32_t io = 0; io < 4; io++) {
        int32_t il = (ii * 4 + io) + 0 / gridX;
        int32_t i = in * ((B1_dimension + (gridX - 1)) / gridX) + il;
        if (i >= B1_dimension)
          continue;

        if (i >= (in + 1) * ((B1_dimension + (gridX - 1)) / gridX - 0 / gridX))
          continue;

        for (int32_t jl = (0 / gridY); jl < ((B2_dimension + (gridY - 1)) / gridY); jl++) {
          int32_t j = jn * ((B2_dimension + (gridY - 1)) / gridY) + jl;
          int32_t jB = i * B2_dimension + j;
          Point<2> A_access_point = Point<2>(i, j);
          if (j >= B2_dimension)
            continue;

          if (j >= (jn + 1) * ((B2_dimension + (gridY - 1)) / gridY - 0 / gridY))
            continue;

          for (int32_t k = 0; k < C1_dimension; k++) {
            Point<3> B_access_point = Point<3>(i, j, k);
            Point<1> C_access_point = Point<1>(k);
            A_vals[A_access_point] = A_vals[A_access_point] + B_vals[B_access_point] * C_vals[C_access_point];
          }
        }
      }
    }
    else {
      #pragma clang loop interleave(enable) vectorize(enable)
      for (int32_t io = 0; io < 4; io++) {
        int32_t il = (ii * 4 + io) + 0 / gridX;
        int32_t i = in * ((B1_dimension + (gridX - 1)) / gridX) + il;
        if (i >= (in + 1) * ((B1_dimension + (gridX - 1)) / gridX - 0 / gridX))
          continue;

        for (int32_t jl = (0 / gridY); jl < ((B2_dimension + (gridY - 1)) / gridY); jl++) {
          int32_t j = jn * ((B2_dimension + (gridY - 1)) / gridY) + jl;
          int32_t jB = i * B2_dimension + j;
          Point<2> A_access_point = Point<2>(i, j);
          if (j >= (jn + 1) * ((B2_dimension + (gridY - 1)) / gridY - 0 / gridY))
            continue;

          for (int32_t k = 0; k < C1_dimension; k++) {
            Point<3> B_access_point = Point<3>(i, j, k);
            Point<1> C_access_point = Point<1>(k);
            A_vals[A_access_point] = A_vals[A_access_point] + B_vals[B_access_point] * C_vals[C_access_point];
          }
        }
      }
    }
  }
}

void computeLegion(Context ctx, Runtime* runtime, LogicalRegion A, LogicalRegion B, LogicalRegion C, int32_t gridX, int32_t gridY) {
  int A1_dimension = runtime->get_index_space_domain(get_index_space(A)).hi()[0] + 1;
  int A2_dimension = runtime->get_index_space_domain(get_index_space(A)).hi()[1] + 1;
  auto A_index_space = get_index_space(A);
  int B1_dimension = runtime->get_index_space_domain(get_index_space(B)).hi()[0] + 1;
  int B2_dimension = runtime->get_index_space_domain(get_index_space(B)).hi()[1] + 1;
  int B3_dimension = runtime->get_index_space_domain(get_index_space(B)).hi()[2] + 1;
  auto B_index_space = get_index_space(B);
  int C1_dimension = runtime->get_index_space_domain(get_index_space(C)).hi()[0] + 1;
  auto C_index_space = get_index_space(C);

  Point<2> lowerBound = Point<2>(0, 0);
  Point<2> upperBound = Point<2>((gridX - 1), (gridY - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<2>(lowerBound, upperBound));
  DomainT<2> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<2>(distFusedIndexSpace));
  auto ADomain = runtime->get_index_space_domain(ctx, A_index_space);
  auto BDomain = runtime->get_index_space_domain(ctx, B_index_space);
  auto CDomain = runtime->get_index_space_domain(ctx, C_index_space);
  DomainPointColoring AColoring = DomainPointColoring();
  DomainPointColoring BColoring = DomainPointColoring();
  DomainPointColoring CColoring = DomainPointColoring();
  for (PointInDomainIterator<2> itr = PointInDomainIterator<2>(domain); itr.valid(); itr++) {
    int32_t in = (*itr)[0];
    int32_t jn = (*itr)[1];
    Point<2> AStart = Point<2>((in * ((B1_dimension + (gridX - 1)) / gridX) + 0 / gridX), (jn * ((B2_dimension + (gridY - 1)) / gridY) + 0 / gridY));
    Point<2> AEnd = Point<2>(TACO_MIN((in * ((B1_dimension + (gridX - 1)) / gridX) + ((B1_dimension + (gridX - 1)) / gridX - 1)), ADomain.hi()[0]), TACO_MIN((jn * ((B2_dimension + (gridY - 1)) / gridY) + ((B2_dimension + (gridY - 1)) / gridY - 1)), ADomain.hi()[1]));
    Rect<2> ARect = Rect<2>(AStart, AEnd);
    if (!ADomain.contains(ARect.lo) || !ADomain.contains(ARect.hi)) {
      ARect = ARect.make_empty();
    }
    AColoring[(*itr)] = ARect;
    Point<3> BStart = Point<3>((in * ((B1_dimension + (gridX - 1)) / gridX) + 0 / gridX), (jn * ((B2_dimension + (gridY - 1)) / gridY) + 0 / gridY), 0);
    Point<3> BEnd = Point<3>(TACO_MIN((in * ((B1_dimension + (gridX - 1)) / gridX) + ((B1_dimension + (gridX - 1)) / gridX - 1)), BDomain.hi()[0]), TACO_MIN((jn * ((B2_dimension + (gridY - 1)) / gridY) + ((B2_dimension + (gridY - 1)) / gridY - 1)), BDomain.hi()[1]), TACO_MIN(C1_dimension, BDomain.hi()[2]));
    Rect<3> BRect = Rect<3>(BStart, BEnd);
    if (!BDomain.contains(BRect.lo) || !BDomain.contains(BRect.hi)) {
      BRect = BRect.make_empty();
    }
    BColoring[(*itr)] = BRect;
    Point<1> CStart = Point<1>(0);
    Point<1> CEnd = Point<1>(TACO_MIN(C1_dimension, CDomain.hi()[0]));
    Rect<1> CRect = Rect<1>(CStart, CEnd);
    if (!CDomain.contains(CRect.lo) || !CDomain.contains(CRect.hi)) {
      CRect = CRect.make_empty();
    }
    CColoring[(*itr)] = CRect;
  }
  auto APartition = runtime->create_index_partition(ctx, A_index_space, domain, AColoring, LEGION_DISJOINT_COMPLETE_KIND);
  auto BPartition = runtime->create_index_partition(ctx, B_index_space, domain, BColoring, LEGION_ALIASED_COMPLETE_KIND);
  auto CPartition = runtime->create_index_partition(ctx, C_index_space, domain, CColoring, LEGION_ALIASED_COMPLETE_KIND);
  LogicalPartition ALogicalPartition = runtime->get_logical_partition(ctx, get_logical_region(A), APartition);
  RegionRequirement AReq = RegionRequirement(ALogicalPartition, 0, READ_WRITE, EXCLUSIVE, get_logical_region(A));
  AReq.add_field(FID_VAL);
  LogicalPartition BLogicalPartition = runtime->get_logical_partition(ctx, get_logical_region(B), BPartition);
  RegionRequirement BReq = RegionRequirement(BLogicalPartition, 0, READ_ONLY, EXCLUSIVE, get_logical_region(B));
  BReq.add_field(FID_VAL);
  LogicalPartition CLogicalPartition = runtime->get_logical_partition(ctx, get_logical_region(C), CPartition);
  RegionRequirement CReq = RegionRequirement(CLogicalPartition, 0, READ_ONLY, EXCLUSIVE, get_logical_region(C));
  CReq.add_field(FID_VAL);
  task_4Args taskArgsRaw;
  taskArgsRaw.A1_dimension = A1_dimension;
  taskArgsRaw.A2_dimension = A2_dimension;
  taskArgsRaw.B1_dimension = B1_dimension;
  taskArgsRaw.B2_dimension = B2_dimension;
  taskArgsRaw.B3_dimension = B3_dimension;
  taskArgsRaw.C1_dimension = C1_dimension;
  taskArgsRaw.gridX = gridX;
  taskArgsRaw.gridY = gridY;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_4Args));
  IndexLauncher launcher = IndexLauncher(taskID(4), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(AReq);
  launcher.add_region_requirement(BReq);
  launcher.add_region_requirement(CReq);
  launcher.tag |= TACOMapper::UNTRACK_VALID_REGIONS;
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
  {
    TaskVariantRegistrar registrar(taskID(2), "task_2");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<task_2>(registrar, "task_2");
  }
  {
    TaskVariantRegistrar registrar(taskID(3), "task_3");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<task_3>(registrar, "task_3");
  }
  {
    TaskVariantRegistrar registrar(taskID(4), "task_4");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<task_4>(registrar, "task_4");
  }
}
