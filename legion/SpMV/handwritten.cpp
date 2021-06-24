#include "legion.h"

using namespace Legion;

enum FieldIDs {
  FID_VALUE,
  FID_INDEX,
  FID_RECT_1,
};

enum TaskIDs {
  TID_TOP_LEVEL,
  TID_PARTITION_A2_CRD,
  TID_SPMV,
  TID_SPMV_POS_SPLIT,
};

struct spmvArgs {
  int n;
  int pieces;
};

struct spmvPosSplitArgs {
  int nnz;
};

template<typename T>
void allocate_fields(Legion::Context ctx, Legion::Runtime* runtime, Legion::FieldSpace valSpace, Legion::FieldID fid, std::string name) {
  Legion::FieldAllocator allocator = runtime->create_field_allocator(ctx, valSpace);
  allocator.allocate_field(sizeof(T), fid);
  runtime->attach_name(valSpace, fid, name.c_str());
}

Legion::PhysicalRegion getRegionToWrite(Legion::Context ctx, Legion::Runtime* runtime, Legion::LogicalRegion r, Legion::LogicalRegion parent, FieldID fid) {
  Legion::RegionRequirement req(r, READ_WRITE, EXCLUSIVE, parent);
  req.add_field(fid);
  return runtime->map_region(ctx, req);
}

void top_level_task(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  auto args = runtime->get_input_args();
  bool posSplit = false;
  for (int i = 1; i < args.argc; i++) {
    if (strcmp(args.argv[i], "-pos") == 0) {
      posSplit = true;
      continue;
    }
  }

  int n = 10;
  int m = 10;
  int nnz = 20;

  auto valFSpace = runtime->create_field_space(ctx);
  allocate_fields<double>(ctx, runtime, valFSpace, FID_VALUE, "vals");
  auto idxFSpace = runtime->create_field_space(ctx);
  allocate_fields<int32_t>(ctx, runtime, idxFSpace, FID_INDEX, "idxs");
  auto rFSpace = runtime->create_field_space(ctx);
  allocate_fields<Rect<1>>(ctx, runtime, rFSpace, FID_RECT_1, "rects");

  auto yIspace = runtime->create_index_space(ctx, Rect<1>(0, n - 1));
  auto xIspace = runtime->create_index_space(ctx, Rect<1>(0, m - 1));

  auto A2_pos_ispace = runtime->create_index_space(ctx, Rect<1>(0, n - 1));
  auto A2_crd_ispace = runtime->create_index_space(ctx, Rect<1>(0, nnz - 1));
  auto A_vals_ispace = runtime->create_index_space(ctx, Rect<1>(0, nnz - 1));

  auto y = runtime->create_logical_region(ctx, yIspace, valFSpace);
  auto x = runtime->create_logical_region(ctx, xIspace, valFSpace);
  auto A2_pos = runtime->create_logical_region(ctx, A2_pos_ispace, rFSpace);
  auto A2_crd = runtime->create_logical_region(ctx, A2_crd_ispace, idxFSpace);
  auto A_vals = runtime->create_logical_region(ctx, A_vals_ispace, valFSpace);

  // Fill all of the fields. Somehow find a parallel method to constructing these tensors.
  // TODO (rohany): I think such a method has to be parallel, as there isn't a good way otherwise
  //  to ensure that each node only reads a subset of the data and assembles it.
  runtime->fill_field<double>(ctx, y, y, FID_VALUE, 0);
  runtime->fill_field<double>(ctx, x, x, FID_VALUE, 0);
  runtime->fill_field<Rect<1>>(ctx, A2_pos, A2_pos, FID_RECT_1, Rect<1>{0, 0});
  runtime->fill_field<int32_t>(ctx, A2_crd, A2_crd, FID_INDEX, 0);
  runtime->fill_field<double>(ctx, A_vals, A_vals, FID_VALUE, 0);
  {
    auto xreg = getRegionToWrite(ctx, runtime, x, x, FID_VALUE);
    FieldAccessor<READ_WRITE,double,1,coord_t, Realm::AffineAccessor<double, 1, coord_t>> xrw(xreg, FID_VALUE);
    for (int i = 0; i < m; i++) {
      xrw[i] = i;
    }
    runtime->unmap_region(ctx, xreg);
  }
  {
    std::vector<int32_t> a2pos = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20};
    auto a2posreg = getRegionToWrite(ctx, runtime, A2_pos, A2_pos, FID_RECT_1);
    FieldAccessor<READ_WRITE, Rect<1> ,1,coord_t, Realm::AffineAccessor<Rect<1>, 1, coord_t>> a2posrw(a2posreg, FID_RECT_1);
    for (int i = 0; i < n; i++) {
      a2posrw[i] = Rect<1>(a2pos[i], a2pos[i+1] - 1);
    }
    runtime->unmap_region(ctx, a2posreg);
  }
  {
    std::vector<int32_t> a2crd = {0, 5, 0, 5, 0, 5, 0, 5, 0, 5, 0, 5, 0, 5, 0, 5, 0, 5, 0, 5};
    auto a2crdreg = getRegionToWrite(ctx, runtime, A2_crd, A2_crd, FID_INDEX);
    FieldAccessor<READ_WRITE,int32_t,1,coord_t, Realm::AffineAccessor<int32_t, 1, coord_t>> a2crdrw(a2crdreg, FID_INDEX);
    for (int i = 0; i < nnz; i++) {
      a2crdrw[i] = a2crd[i];
    }
    runtime->unmap_region(ctx, a2crdreg);
  }
  {
    std::vector<int32_t> avals = {0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9};
    auto avalsreg = getRegionToWrite(ctx, runtime, A_vals, A_vals, FID_VALUE);
    FieldAccessor<READ_WRITE,double,1,coord_t, Realm::AffineAccessor<double, 1, coord_t>> avalsrw(avalsreg, FID_VALUE);
    for (int i = 0; i < nnz; i++) {
      avalsrw[i] = avals[i];
    }
    runtime->unmap_region(ctx, avalsreg);
  }

  auto pieces = 4;
  auto domain = Domain(Rect<1>(0, pieces - 1));

  // Do a position split -> equal partition of non-zeros.
  if (posSplit) {
    // In this case, we want to start with a partition of the crd/values arrays.
    DomainPointColoring A2_crd_col, A_vals_col, y_col;
    for (PointInDomainIterator<1> itr(domain); itr(); itr++) {
      int fpos0 = *itr;
      int start = fpos0 * ((nnz + pieces - 1) / pieces);
      int end = fpos0 * ((nnz + pieces - 1) / pieces) + ((nnz + pieces - 1) / pieces) - 1;
      A2_crd_col[*itr] = Rect<1>(start, end);
      A_vals_col[*itr] = Rect<1>(start, end);
    }
    auto A_vals_partition = runtime->create_index_partition(ctx, A_vals_ispace, domain, A_vals_col, LEGION_DISJOINT_COMPLETE_KIND);
    auto A_vals_logical_partition = runtime->get_logical_partition(ctx, A_vals, A_vals_partition);
    auto A2_crd_partition = runtime->create_index_partition(ctx, A2_crd_ispace, domain, A2_crd_col, LEGION_DISJOINT_COMPLETE_KIND);
    auto A2_crd_logical_partition = runtime->get_logical_partition(ctx, A2_crd, A2_crd_partition);

    // Now use the partition of A2_crd to partition A2_pos.
    auto A2_pos_part = runtime->create_partition_by_preimage_range(
      ctx,
      A2_crd_partition,
      A2_pos,
      A2_pos,
      FID_RECT_1,
      runtime->get_index_partition_color_space_name(A2_crd_partition)
    );
    auto A2_pos_logical_partition = runtime->get_logical_partition(ctx, A2_pos, A2_pos_part);

    // Use the partition bounds on A2_pos_logical_partition to partition y.
    for (PointInDomainIterator<1> itr(domain); itr(); itr++) {
      auto subreg = runtime->get_logical_subregion_by_color(ctx, A2_pos_logical_partition, DomainPoint(*itr));
      auto dom = runtime->get_index_space_domain(ctx, subreg.get_index_space());
      y_col[*itr] = Rect<1>(dom.lo(), dom.hi());
    }
    auto y_partition = runtime->create_index_partition(ctx, yIspace, domain, y_col, LEGION_ALIASED_COMPLETE_KIND);
    auto y_logical_partition = runtime->get_logical_partition(ctx, y, y_partition);

    RegionRequirement yReq = RegionRequirement(y_logical_partition, 0, LEGION_REDOP_SUM_FLOAT64, SIMULTANEOUS, y).add_field(FID_VALUE);
    RegionRequirement A2_pos_req = RegionRequirement(A2_pos_logical_partition, 0, READ_ONLY, EXCLUSIVE, A2_pos).add_field(FID_RECT_1);
    RegionRequirement A2_crd_req = RegionRequirement(A2_crd_logical_partition, 0, READ_ONLY, EXCLUSIVE, A2_crd).add_field(FID_INDEX);
    RegionRequirement A_vals_req = RegionRequirement(A_vals_logical_partition, 0, READ_ONLY, EXCLUSIVE, A_vals).add_field(FID_VALUE);
    RegionRequirement xReq = RegionRequirement(x, READ_ONLY, EXCLUSIVE, x).add_field(FID_VALUE);
    spmvPosSplitArgs args;
    args.nnz = nnz;
    IndexLauncher launcher = IndexLauncher(TID_SPMV_POS_SPLIT, domain, TaskArgument(&args, sizeof(spmvPosSplitArgs)), ArgumentMap());
    launcher.add_region_requirement(yReq);
    launcher.add_region_requirement(A2_pos_req);
    launcher.add_region_requirement(A2_crd_req);
    launcher.add_region_requirement(A_vals_req);
    launcher.add_region_requirement(xReq);
    auto fm = runtime->execute_index_space(ctx, launcher);
    fm.wait_all_results();
  } else {
    // Do a partition across i.

    // Partition y.
    DomainPointColoring yCol;
    for (PointInDomainIterator<1> itr(domain); itr(); itr++) {
      int i = (*itr)[0];
      Point<1> yStart = i * ((n + (pieces - 1)) / pieces);
      Point<1> yEnd = i * ((n + (pieces - 1)) / pieces) + ((n + (pieces - 1)) / pieces) - 1;
      Rect<1> yRect = Rect<1>(yStart, yEnd);
      yCol[*itr] = yRect;
    }
    auto yPartition = runtime->create_index_partition(ctx, yIspace, domain, yCol, LEGION_DISJOINT_COMPLETE_KIND);
    auto yLogicalPartition = runtime->get_logical_partition(ctx, y, yPartition);

    // x isn't getting partitioned.

    // Partition A.
    DomainPointColoring A2_pos_col, A2_crd_col, A_vals_col;
    for (PointInDomainIterator<1> itr(domain); itr(); itr++) {
      // Lower bound of A1 = i * ((n + (pieces - 1)) / pieces).
      // Upper bound of A1 = i * ((n + (pieces - 1)) / pieces) + ((n + (pieces - 1)) / pieces) - 1.
      // So now, I can partition A2_pos using these values.
      int i = (*itr)[0];
      Point<1> A2_pos_start = i * ((n + (pieces - 1)) / pieces);
      Point<1> A2_pos_end = i * ((n + (pieces - 1)) / pieces) + ((n + (pieces - 1)) / pieces) - 1;
      Rect<1> A2_pos_rect = Rect<1>(A2_pos_start, A2_pos_end);
      A2_pos_col[*itr] = A2_pos_rect;
    }
    auto A2_pos_partition = runtime->create_index_partition(ctx, A2_pos_ispace, domain, A2_pos_col, LEGION_DISJOINT_COMPLETE_KIND);
    auto A2_pos_logical_partition = runtime->get_logical_partition(ctx, A2_pos, A2_pos_partition);

    // TODO (rohany): This has to be mapped in the same way as the loop is meant to be mapped.
    auto A2_crd_part = runtime->create_partition_by_image_range(
        ctx,
        A2_crd_ispace,
        A2_pos_logical_partition,
        A2_pos,
        FID_RECT_1,
        runtime->get_index_partition_color_space_name(ctx, A2_pos_partition)
    );
    auto A2_crd_logical_partition = runtime->get_logical_partition(ctx, A2_crd, A2_crd_part);

    // Use the crd partition to make a partition of the values.
    for (PointInDomainIterator<1> itr(domain); itr(); itr++) {
      auto subreg = runtime->get_logical_subregion_by_color(ctx, A2_crd_logical_partition, DomainPoint(*itr));
      auto subregIspace = subreg.get_index_space();
      auto bounds = runtime->get_index_space_domain(subregIspace);
      A_vals_col[*itr] = Rect<1>(bounds.lo(), bounds.hi());
    }
    auto A_vals_partition = runtime->create_index_partition(ctx, A_vals_ispace, domain, A_vals_col, LEGION_DISJOINT_COMPLETE_KIND);
    auto A_vals_logical_partition = runtime->get_logical_partition(ctx, A_vals, A_vals_partition);

    RegionRequirement yReq = RegionRequirement(yLogicalPartition, 0, READ_WRITE, EXCLUSIVE, y).add_field(FID_VALUE);
    RegionRequirement A2_pos_req = RegionRequirement(A2_pos_logical_partition, 0, READ_ONLY, EXCLUSIVE, A2_pos).add_field(FID_RECT_1);
    RegionRequirement A2_crd_req = RegionRequirement(A2_crd_logical_partition, 0, READ_ONLY, EXCLUSIVE, A2_crd).add_field(FID_INDEX);
    RegionRequirement A_vals_req = RegionRequirement(A_vals_logical_partition, 0, READ_ONLY, EXCLUSIVE, A_vals).add_field(FID_VALUE);
    RegionRequirement xReq = RegionRequirement(x, READ_ONLY, EXCLUSIVE, x).add_field(FID_VALUE);
    spmvArgs args;
    args.n = n; args.pieces = pieces;
    IndexLauncher launcher = IndexLauncher(TID_SPMV, domain, TaskArgument(&args, sizeof(spmvArgs)), ArgumentMap());
    launcher.add_region_requirement(yReq);
    launcher.add_region_requirement(A2_pos_req);
    launcher.add_region_requirement(A2_crd_req);
    launcher.add_region_requirement(A_vals_req);
    launcher.add_region_requirement(xReq);
    auto fm = runtime->execute_index_space(ctx, launcher);
    fm.wait_all_results();
  }

  {
    // Now the bounds of A2_crd are dependent on the _values_ that are in the A2_pos partitions.
    // TODO (rohany): The obvious way to do this is to map the individual partitions of A2_pos
    //  and then use those bounds for the next level. However, this means that every node needs
    //  to read A2_pos. A maybe smarter way to do this is to do an index launch so that each node
    //  gets only a slice of A2_pos to look at and to make their partition. Then, they can return
    //  a rectangle that contains the values of their point's bounds for A2_crd.
    // TODO (rohany): This is just SpMV, but I could imagine doing partitioning in parallel like this
    //  with index launches for dimensions past this, and pass the future map to those calls?
    //  [Thoughts]: I don't actually know if that will work. The level then needs the whole next
    //  level to be partitioned appropriately, so we need to wait on the results to come back. So
    //  maybe it's not possible to pipeline this.
    // TODO (rohany): Need to ask alex about a legion-ic way to do this, since it's very similar to
    //  dependent partitioning. If I can't think use dependent partitioning directly, I want to be
    //  able at least to mimic how it is implemented (in a non-blocking way).
    // RegionRequirement A2_pos_req = RegionRequirement(A2_pos_logical_partition, 0, READ_ONLY, EXCLUSIVE, A2_pos);
    // A2_pos_req.add_field(FID_INDEX);
    // IndexLauncher launcher = IndexLauncher(TID_PARTITION_A2_CRD, domain, TaskArgument(), ArgumentMap());
    // launcher.add_region_requirement(A2_pos_req);
    // auto fm = runtime->execute_index_space(ctx, launcher);
    // fm.wait_all_results();

    // for (PointInDomainIterator<1> itr(domain); itr(); itr++) {
    //   A_vals_col[*itr] = fm[(*itr)].get<Rect<1>>();
    //   A2_crd_col[*itr] = fm[(*itr)].get<Rect<1>>();
    // }

    // auto A2_crd_partition = runtime->create_index_partition(ctx, A2_crd_ispace, domain, A2_crd_col, LEGION_DISJOINT_COMPLETE_KIND);
    // auto A2_crd_logical_partition = runtime->get_logical_partition(ctx, A2_crd, A2_crd_partition);
    // auto A_vals_partition = runtime->create_index_partition(ctx, A_vals_ispace, domain, A_vals_col, LEGION_DISJOINT_COMPLETE_KIND);
    // auto A_vals_logical_partition = runtime->get_logical_partition(ctx, A_vals, A_vals_partition);

    // This is a more manual way of performing the dependent partitioning operation
    // that is actually more efficient (big-O wise).
  }

  {
    auto yreg = getRegionToWrite(ctx, runtime, y, y, FID_VALUE);
    FieldAccessor<READ_WRITE,double,1,coord_t, Realm::AffineAccessor<double, 1, coord_t>> yrw(yreg, FID_VALUE);
    for (int i = 0; i < n; i++) {
      std::cout << yrw[i] << " ";
    }
    std::cout << std::endl;
    runtime->unmap_region(ctx, yreg);
  }
}

Rect<1> partition_A2_crd(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  auto A2_pos_rg = regions[0];
  typedef FieldAccessor<READ_ONLY,int32_t,1,coord_t,Realm::AffineAccessor<int32_t,1,coord_t>> Accessor;
  Accessor A2_accessor(A2_pos_rg, FID_INDEX);
  auto A2_index_space = A2_pos_rg.get_logical_region().get_index_space();
  auto domain = runtime->get_index_space_domain(ctx, A2_index_space);
  // Assuming I have pieces and n here, I would actually do the lower and upper bounds of A2_pos itself.
  auto lo = A2_accessor[domain.lo()];
  auto hi = A2_accessor[domain.hi()] - 1;
  return {lo, hi};
}

void spmv(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  spmvArgs* args = (spmvArgs*)(task->args);
  int in = task->index_point[0];
  int n = args->n;
  int pieces = args->pieces;

  typedef FieldAccessor<READ_ONLY,int32_t,1,coord_t,Realm::AffineAccessor<int32_t,1,coord_t>> AccessorI;
  typedef FieldAccessor<READ_ONLY,Rect<1>,1,coord_t,Realm::AffineAccessor<Rect<1>,1,coord_t>> AccessorR;
  typedef FieldAccessor<READ_ONLY,double,1,coord_t,Realm::AffineAccessor<double,1,coord_t>> AccessorD;
  typedef FieldAccessor<READ_WRITE,double,1,coord_t,Realm::AffineAccessor<double,1,coord_t>> AccessorWD;
  AccessorWD y_vals(regions[0], FID_VALUE);
  AccessorR A2_pos(regions[1], FID_RECT_1);
  AccessorI A2_crd(regions[2], FID_INDEX);
  AccessorD A_vals(regions[3], FID_VALUE);
  AccessorD x_vals(regions[4], FID_VALUE);

  for (int32_t i = in * ((n + (pieces - 1)) / pieces); i < i * ((n + (pieces - 1)) / pieces) + ((n + (pieces - 1)) / pieces) - 1; i++) {
    if (i >= n) {
      continue;
    }
    if (i >= (in + 1) * ((n + 3) / 4)) {
      continue;
    }

    double tjy_val = 0.0;
    for (int32_t jA = A2_pos[i].lo; jA <= A2_pos[i].hi; jA++) {
      int32_t j = A2_crd[jA];
      tjy_val += A_vals[jA] * x_vals[j];
    }
    y_vals[i] = tjy_val;
  }
}

void spmvPosSplit(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  spmvPosSplitArgs* args = (spmvPosSplitArgs*)(task->args);
  int nnz = args->nnz;
  int fpos0 = task->index_point[0];

  typedef ReductionAccessor<SumReduction<double>,true,1,coord_t,Realm::AffineAccessor<double,1,coord_t>> AccessorReducedouble2;
  typedef FieldAccessor<READ_ONLY,int32_t,1,coord_t,Realm::AffineAccessor<int32_t,1,coord_t>> AccessorI;
  typedef FieldAccessor<READ_ONLY,Rect<1>,1,coord_t,Realm::AffineAccessor<Rect<1>,1,coord_t>> AccessorR;
  typedef FieldAccessor<READ_ONLY,double,1,coord_t,Realm::AffineAccessor<double,1,coord_t>> AccessorD;
  AccessorReducedouble2 y_vals(regions[0], FID_VALUE, LEGION_REDOP_SUM_FLOAT64);
  AccessorR A2_pos(regions[1], FID_RECT_1);
  AccessorI A2_crd(regions[2], FID_INDEX);
  AccessorD A_vals(regions[3], FID_VALUE);
  AccessorD x_vals(regions[4], FID_VALUE);

  auto posDom = runtime->get_index_space_domain(regions[1].get_logical_region().get_index_space());
  // Instead of the binary search, use the partition's boundaries.
  int32_t i_pos = posDom.lo()[0];
  int32_t i = i_pos;

  for (int32_t fpos1 = 0; fpos1 < ((nnz + 3) / 4); fpos1++) {
    int32_t fposA = fpos0 * (nnz / 4) + fpos1;
    if (fposA >= (fpos0 + 1) * ((nnz + 3) / 4))
      continue;
    if (fposA >= nnz)
      continue;

    int32_t f = A2_crd[fposA];
    // TODO (rohany): I think that this needs to be a +1 cuz the points are inclusive.
    while (fposA == (A2_pos[i_pos].hi + 1)) {
      i_pos++;
      i = i_pos;
    }
    y_vals[i] <<= A_vals[fposA] * x_vals[f];
  }
}


int main(int argc, char** argv) {
  Runtime::set_top_level_task_id(TID_TOP_LEVEL);
  {
    TaskVariantRegistrar registrar(TID_TOP_LEVEL, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_replicable();
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }
  {
    TaskVariantRegistrar registrar(TID_PARTITION_A2_CRD, "partition_a2_crd");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<Rect<1>, partition_A2_crd>(registrar, "partition_a2_crd");
  }
  {
    TaskVariantRegistrar registrar(TID_SPMV, "spmv");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<spmv>(registrar, "spmv");
  }
  {
    TaskVariantRegistrar registrar(TID_SPMV_POS_SPLIT, "spmvPos");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<spmvPosSplit>(registrar, "spmvPos");
  }
  return Runtime::start(argc, argv);
}