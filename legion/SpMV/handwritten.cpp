#include "legion.h"
#include "realm/cmdline.h"
#include "mappers/default_mapper.h"

#include "legion/legion_utilities.h"

#include "hdf5_utils.h"

using namespace Legion;

enum FieldIDs {
  FID_VALUE,
  FID_INDEX,
  FID_RECT_1,
  FID_COORD_X,
  FID_COORD_Y,
};

enum TaskIDs {
  TID_TOP_LEVEL,
  TID_PARTITION_A2_CRD,
  TID_SPMV,
  TID_SPMV_POS_SPLIT,
  TID_PRINT_COORD_LIST,
  TID_PACK_A_CSR,
};

struct spmvArgs {
  int n;
  int pieces;
};

struct spmvPosSplitArgs {
  int nnz;
};


struct LegionTensor {
  int32_t order;
  std::vector<int32_t> dims;

  std::vector<std::vector<LogicalRegion>> indices;
  std::vector<std::vector<LogicalRegion>> indicesParents;

  LogicalRegion vals;
  LogicalRegion valsParent;

  // Has some sort of serialize method, deserialize to add dims and regions
  // to a task launcher.

  void serialize(Serializer& sez) {
    sez.serialize(this->order);
    for (auto dim : this->dims) {
      sez.serialize(dim);
    }
  }

  void deserialize(Deserializer& derez) {
    derez.deserialize(this->order);
    for (int32_t i = 0; i < this->order; i++) {
      int32_t val;
      derez.deserialize(val);
      this->dims.push_back(val);
    }
  }
};

struct LegionTensorPartition {
  LogicalPartition** indicesPartitions;
  LogicalPartition valsPartition;
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

void printCoordListMatrix(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  auto reg = regions[0];
  // Declare some accessors.
  FieldAccessor<READ_ONLY,int32_t,1,coord_t, Realm::AffineAccessor<int32_t, 1, coord_t>> x(reg, FID_COORD_X);
  FieldAccessor<READ_ONLY,int32_t,1,coord_t, Realm::AffineAccessor<int32_t, 1, coord_t>> y(reg, FID_COORD_Y);
  FieldAccessor<READ_ONLY,double,1,coord_t, Realm::AffineAccessor<double, 1, coord_t>> v(reg, FID_VALUE);

  for (PointInDomainIterator<1> itr(reg); itr(); itr++) {
    int i = *itr;
    printf("%d %d %lf\n", x[i], y[i], v[i]);
  }
}

LogicalRegion getSubRegion(Context ctx, Runtime* runtime, LogicalRegion region, size_t numElems) {
  // Get a partition of the region of the appropriate size.
  auto bounds = Rect<1>(0, numElems - 1);
  IndexSpaceT<1> colorSpace = runtime->create_index_space(ctx, Rect<1>(0, 0));
  Transform<1,1> transform;
  transform[0][0] = 0;
  auto ip = runtime->create_partition_by_restriction(
      ctx,
      region.get_index_space(),
      colorSpace,
      transform,
      bounds,
      DISJOINT_KIND
      // Do I need to pick a color for this partition?
  );
  auto lp = runtime->get_logical_partition(ctx, region, ip);
  return runtime->get_logical_subregion_by_color(ctx, lp, 0);
}

PhysicalRegion lgMalloc(Context ctx, Runtime* runtime, LogicalRegion region, size_t numElems, FieldID fid) {
  auto subreg = getSubRegion(ctx, runtime, region, numElems);
  RegionRequirement req(subreg, READ_WRITE, EXCLUSIVE, region, Mapping::DefaultMapper::EXACT_REGION);
  req.add_field(fid);
  return runtime->map_region(ctx, req);
}

// Interestingly, with Legion, I don't have to realloc like this. Instead, I can just ask Legion
// for the next x elements, rather than mapping twice the space! It's going to hold on to the
// data that I've already written, and I don't need to do offset arithmetic calculations either.
PhysicalRegion lgRealloc(Context ctx, Runtime* runtime, LogicalRegion region, PhysicalRegion pg, size_t numElems, FieldID fid) {
  runtime->unmap_region(ctx, pg);
  return lgMalloc(ctx, runtime, region, numElems, fid);
}

struct PackArgs {
  size_t A1_dimension;
};

void packACSR(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  typedef FieldAccessor<READ_ONLY,int32_t,1,coord_t,Realm::AffineAccessor<int32_t,1,coord_t>> RAccessorI;
  typedef FieldAccessor<READ_ONLY,double,1,coord_t,Realm::AffineAccessor<double,1,coord_t>> RAccessorD;

  typedef FieldAccessor<READ_WRITE,int32_t,1,coord_t,Realm::AffineAccessor<int32_t,1,coord_t>> AccessorI;
  typedef FieldAccessor<READ_WRITE,Rect<1>,1,coord_t,Realm::AffineAccessor<Rect<1>,1,coord_t>> AccessorR;
  typedef FieldAccessor<READ_WRITE,double,1,coord_t,Realm::AffineAccessor<double,1,coord_t>> AccessorD;

  LegionTensor coordList;
  Deserializer derez(task->args, task->arglen);
  coordList.deserialize(derez);

  int32_t A1_dimension = coordList.dims[0];

  // These are physically mapped.
  RAccessorI COO1(regions[0], FID_INDEX);
  RAccessorI COO2(regions[1], FID_INDEX);
  RAccessorD COOVals(regions[2], FID_VALUE);

  auto nnz = Rect<1>(runtime->get_index_space_domain(regions[0].get_logical_region().get_index_space())).hi + 1;

  // These regions are all virtually mapped. We'll have to malloc / realloc them.
  auto lPos = regions[3].get_logical_region();
  auto lCrd = regions[4].get_logical_region();
  auto lVals = regions[5].get_logical_region();

  // Initialize A2_pos.
  auto A2_pos_phys = lgMalloc(ctx, runtime, lPos, A1_dimension, FID_RECT_1);
  AccessorR A2_pos(A2_pos_phys, FID_RECT_1);
  for (int i = 0; i < A1_dimension; i++) {
    // Initialize each rect as empty.
    A2_pos[i] = Rect<1>(0, -1);
  }

  // The default starting size. Can bump this up for later.
  int32_t A_capacity = 100;
  int32_t jA = 0;
  auto A_vals_phys = lgMalloc(ctx, runtime, lVals, A_capacity, FID_VALUE);
  AccessorD A_vals(A_vals_phys, FID_VALUE);

  int32_t A2_crd_size = 100;
  auto A2_crd_phys = lgMalloc(ctx, runtime, lCrd, A2_crd_size, FID_INDEX);
  AccessorI A2_crd(A2_crd_phys, FID_INDEX);

  int32_t iA = 0;
  // First level loop.
  while (iA < nnz) {
    int32_t i = COO1[iA];
    int32_t A1_segend = iA + 1;
    // Walk through all coordinates equal to the current to find the bounds of
    // the segment.
    while (A1_segend < nnz && COO1[A1_segend] == i) {
      A1_segend++;
    }

    int32_t pA2_begin = jA;

    int32_t jA_COO = iA;
    // Second level loop.
    while (jA_COO < A1_segend) {
      int32_t j = COO2[jA_COO];
      double A_COO_val = COOVals[jA_COO];
      jA_COO++;
      while (jA_COO < A1_segend && COO2[jA_COO] == j) {
        A_COO_val += COOVals[jA_COO];
        jA_COO++;
      }
      if (A_capacity <= jA) {
        A_vals_phys = lgRealloc(ctx, runtime, lVals, A_vals_phys, A_capacity * 2, FID_VALUE);
        A_vals = AccessorD(A_vals_phys, FID_VALUE);
        A_capacity *= 2;
      }
      A_vals[jA] = A_COO_val;
      if (A2_crd_size <= jA) {
        A2_crd_phys = lgRealloc(ctx, runtime, lCrd, A2_crd_phys, A2_crd_size * 2, FID_INDEX);
        A2_crd = AccessorI(A2_crd_phys, FID_INDEX);
        A2_crd_size *= 2;
      }
      A2_crd[jA] = j;
      jA++;
    }

    // Minus 1 is important here.
    // A2_pos[i].hi = jA - pA2_begin - 1;
    A2_pos[i].hi = A2_pos[i].hi + (jA - pA2_begin);
    iA = A1_segend;
  }

  // This is different than the standard packing operation for normal sparse levels.
  // Here, we have to do something more than the prefix sum operation. We need to
  // increment each position by the accumulator, and then bump the accumulator with
  // the high value of the rectangle because the bounds are inclusive.
  int32_t acc = 0;
  for (int i = 0; i < A1_dimension; i++) {
    int32_t hi = A2_pos[i].hi;
    A2_pos[i].lo = A2_pos[i].lo + acc;
    A2_pos[i].hi = A2_pos[i].hi + acc;

    // Now increment acc by hi?
    acc += hi + 1;
  }

  // Unmap all physical regions.
  runtime->unmap_region(ctx, A2_crd_phys);
  runtime->unmap_region(ctx, A2_pos_phys);
  runtime->unmap_region(ctx, A_vals_phys);
}

class ExternalResourceCollector {
public:
  ExternalResourceCollector(Context ctx, Runtime* runtime) : ctx(ctx), runtime(runtime) {}
  ~ExternalResourceCollector() {
    // On destruct, free all of the resources.
    for (auto reg : this->physicalRegions) {
      runtime->detach_external_resource(ctx, reg).wait();
    }
  }
  void attach(PhysicalRegion region) {
    this->physicalRegions.push_back(region);
  }
private:
  Context ctx;
  Runtime* runtime;
  std::vector<PhysicalRegion> physicalRegions;
};

LegionTensor loadCoordList(Context ctx, Runtime* runtime, std::string filename, ExternalResourceCollector& col) {
  LegionTensor result;

  // Get out metadata from the file.
  size_t order, nnz;
  getCoordListHDF5Meta(filename, order, nnz);
  result.order = order;

  // Create all of the field and index spaces.
  auto fispace = runtime->create_field_space(ctx);
  auto fvspace = runtime->create_field_space(ctx);
  {
    FieldAllocator fa = runtime->create_field_allocator(ctx, fispace);
    fa.allocate_field(sizeof(int32_t), FID_INDEX);
  }
  {
    FieldAllocator fa = runtime->create_field_allocator(ctx, fvspace);
    fa.allocate_field(sizeof(double), FID_VALUE);
  }
  auto dimIspace = runtime->create_index_space(ctx, Rect<1>(0, order - 1));
  auto coordIspace = runtime->create_index_space(ctx, Rect<1>(0, nnz - 1));

  // Finally make the regions.
  auto dims = runtime->create_logical_region(ctx, dimIspace, fispace);
  auto dimsMem = runtime->create_logical_region(ctx, dimIspace, fispace);
  auto coo1 = runtime->create_logical_region(ctx, coordIspace, fispace);
  auto coo2 = runtime->create_logical_region(ctx, coordIspace, fispace);
  auto vals = runtime->create_logical_region(ctx, coordIspace, fvspace);

  // Attach all of the regions.
  auto dimsDisk = attachHDF5(ctx, runtime, dims, {{FID_INDEX, COODimsField}}, filename);
  col.attach(attachHDF5(ctx, runtime, coo1, {{FID_INDEX, COOCoordsFields[0]}}, filename));
  col.attach(attachHDF5(ctx, runtime, coo2, {{FID_INDEX, COOCoordsFields[1]}}, filename));
  col.attach(attachHDF5(ctx, runtime, vals, {{FID_VALUE, COOValsField}}, filename));

  // Extract the dimensions into a CPU memory instance.
  {
    CopyLauncher cl;
    cl.add_copy_requirements(
      RegionRequirement(dims, READ_ONLY, EXCLUSIVE, dims),
      RegionRequirement(dimsMem, WRITE_DISCARD, EXCLUSIVE, dimsMem)
    );
    cl.add_src_field(0, FID_INDEX); cl.add_dst_field(0, FID_INDEX);
    runtime->issue_copy_operation(ctx, cl);
  }
  {
    // Map the CPU region directly now.
    auto preg = runtime->map_region(
        ctx,
        RegionRequirement(dimsMem, READ_ONLY, EXCLUSIVE, dimsMem).add_field(FID_INDEX)
    );
    FieldAccessor<READ_ONLY,int32_t,1,coord_t, Realm::AffineAccessor<int32_t, 1, coord_t>> acc(preg, FID_INDEX);
    result.dims = std::vector<int32_t>(result.order);
    for (size_t i = 0; i < order; i++) {
      result.dims[i] = acc[i];
    }
    runtime->unmap_region(ctx, preg);
  }

  // Now, construct the resulting tensor.
  result.indices = std::vector<std::vector<LogicalRegion>>(2);
  result.indicesParents = std::vector<std::vector<LogicalRegion>>(2);
  result.indices[0] = {coo1}; result.indicesParents[0] = {coo1};
  result.indices[1] = {coo2}; result.indicesParents[1] = {coo2};

  result.vals = vals;
  result.valsParent = vals;

  runtime->destroy_logical_region(ctx, dims);
  runtime->destroy_logical_region(ctx, dimsMem);

  return result;
}

LegionTensor initCSR(Context ctx, Runtime* runtime) {
  LegionTensor result;
  result.order = 2;
  result.dims = {0, 0};

  auto posspace = runtime->create_field_space(ctx);
  auto crdspace = runtime->create_field_space(ctx);
  auto valspace = runtime->create_field_space(ctx);
  allocate_fields<Rect<1>>(ctx, runtime, posspace, FID_RECT_1, "pos");
  allocate_fields<int32_t>(ctx, runtime, crdspace, FID_INDEX, "crd");
  allocate_fields<double>(ctx, runtime, valspace, FID_VALUE, "vals");

  // Let's try the reallocating logic for all of the arrays.
  auto maxSize = 1 << 30;
  auto ispace = runtime->create_index_space(ctx, Rect<1>(0, maxSize - 1));

  auto pos = runtime->create_logical_region(ctx, ispace, posspace);
  auto crd = runtime->create_logical_region(ctx, ispace, crdspace);
  auto vals = runtime->create_logical_region(ctx, ispace, valspace);

  result.indices = std::vector<std::vector<LogicalRegion>>(result.order);
  result.indicesParents = std::vector<std::vector<LogicalRegion>>(result.order);
  result.indices[1] = {pos, crd};
  result.indicesParents[1] = {pos, crd};

  result.vals = vals;
  result.valsParent = vals;

  // Fill all of the fields before returning.
  runtime->fill_field(ctx, pos, pos, FID_RECT_1, Rect<1>(0, 0));
  runtime->fill_field(ctx, crd, crd, FID_INDEX, int32_t(0));
  runtime->fill_field(ctx, vals, vals, FID_VALUE, double(0));

  return result;
}

void top_level_task(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  bool posSplit = false, dump = false;
  std::string filename;
  Realm::CommandLineParser parser;
  int pieces = 1;
  parser.add_option_bool("-pos", posSplit);
  parser.add_option_string("-file", filename);
  parser.add_option_bool("-dump", dump);
  parser.add_option_int("-pieces", pieces);
  auto args = Runtime::get_input_args();
  assert(parser.parse_command_line(args.argc, args.argv));
  assert(!filename.empty());

  // Load the coordinate list matrix.
  ExternalResourceCollector col(ctx, runtime);
  auto coordList = loadCoordList(ctx, runtime, filename, col);

  size_t nnz = Rect<1>(runtime->get_index_space_domain(coordList.vals.get_index_space())).hi + 1;
  int n = coordList.dims[0];
  int m = coordList.dims[1];

  // Let's print out what is in the matrix.
  // TODO (rohany): Rewrite this.
//  if (dump) {
//    RegionRequirement r(coordList, READ_ONLY, EXCLUSIVE, coordList);
//    r.add_field(FID_COORD_X).add_field(FID_COORD_Y).add_field(FID_VALUE);
//    TaskLauncher launcher(TID_PRINT_COORD_LIST, TaskArgument());
//    launcher.add_region_requirement(r);
//    runtime->execute_task(ctx, launcher).wait();
//  }

  LegionTensor A = initCSR(ctx, runtime);
  // Launch the pack task.
  {
    Serializer s;
    coordList.serialize(s);
    TaskLauncher launcher(TID_PACK_A_CSR, TaskArgument(s.get_buffer(), s.get_used_bytes()));

    // RegionRequirements for the COO matrix.
    launcher.add_region_requirement(
        RegionRequirement(coordList.indices[0][0], READ_ONLY, EXCLUSIVE, coordList.indicesParents[0][0])
            .add_field(FID_INDEX)
    );
    launcher.add_region_requirement(
        RegionRequirement(coordList.indices[1][0], READ_ONLY, EXCLUSIVE, coordList.indicesParents[1][0])
            .add_field(FID_INDEX)
    );
    launcher.add_region_requirement(
        RegionRequirement(coordList.vals, READ_ONLY, EXCLUSIVE, coordList.valsParent)
            .add_field(FID_VALUE)
    );

    // Region requirements for the CSR matrix.
    launcher.add_region_requirement(
        RegionRequirement(A.indices[1][0], READ_WRITE, EXCLUSIVE, A.indicesParents[1][0], Mapping::DefaultMapper::VIRTUAL_MAP)
            .add_field(FID_RECT_1)
    );
    launcher.add_region_requirement(
        RegionRequirement(A.indices[1][1], READ_WRITE, EXCLUSIVE, A.indicesParents[1][1], Mapping::DefaultMapper::VIRTUAL_MAP)
            .add_field(FID_INDEX)
    );
    launcher.add_region_requirement(
        RegionRequirement(A.vals, READ_WRITE, EXCLUSIVE, A.valsParent, Mapping::DefaultMapper::VIRTUAL_MAP)
            .add_field(FID_VALUE)
    );
    runtime->execute_task(ctx, launcher).wait();

    // TODO (rohany): The pack should return new metadata about A that we should use to update
    //  the LegionTensor. I don't know how this works in an environment where we can run ahead...
    //  The API here seems like something I need to talk through with Fred/Alex.
    //  It seems like if we run ahead, we can't keep any state about what is going on in a LegionTensor,
    //  when I would really like to keep information about what are the active subregions etc.
    //  One idea: We can serialize a LegionTensor! The names of the logical regions etc can be passed
    //  around and serialized in futures! This is a bit awkward when maintaining "coherence" around
    //  the parents of each tensor etc, so the constructed results could be aware of that?

    // Use the resulting information to update our LegionTensor.
    A.indices[1][0] = getSubRegion(ctx, runtime, A.indicesParents[1][0], n);
    A.indices[1][1] = getSubRegion(ctx, runtime, A.indicesParents[1][1], nnz);
    A.vals = getSubRegion(ctx, runtime, A.valsParent, nnz);
  }

  // Let's print out the regions and see what happened.
  if (dump) {
    {
      auto a2posreg = lgMalloc(ctx, runtime, A.indicesParents[1][0], n, FID_RECT_1);
      FieldAccessor<READ_WRITE,Rect<1>,1,coord_t, Realm::AffineAccessor<Rect<1>, 1, coord_t>> a2posrw(a2posreg, FID_RECT_1);
      for (int i = 0; i < n; i++) {
        std::cout << a2posrw[i] << " ";
      }
      std::cout << std::endl;
      runtime->unmap_region(ctx, a2posreg);
    }
    {
      auto a2crdreg = lgMalloc(ctx, runtime, A.indicesParents[1][1], nnz, FID_INDEX);
      FieldAccessor<READ_WRITE,int32_t,1,coord_t, Realm::AffineAccessor<int32_t, 1, coord_t>> a2crdrw(a2crdreg, FID_INDEX);
      for (size_t i = 0; i < nnz; i++) {
        std::cout << a2crdrw[i] << " ";
      }
      std::cout << std::endl;
      runtime->unmap_region(ctx, a2crdreg);
    }
    {
      auto avalsreg = lgMalloc(ctx, runtime, A.valsParent, nnz, FID_VALUE);
      FieldAccessor<READ_WRITE,double,1,coord_t, Realm::AffineAccessor<double, 1, coord_t>> avalsrw(avalsreg, FID_VALUE);
      for (size_t i = 0; i < nnz; i++) {
        std::cout << avalsrw[i] << " ";
      }
      std::cout << std::endl;
      runtime->unmap_region(ctx, avalsreg);
    }
  }

  // TODO (rohany): Make these Legion tensors as well.
  // Create x, y and initialize them.
  auto valFSpace = runtime->create_field_space(ctx);
  allocate_fields<double>(ctx, runtime, valFSpace, FID_VALUE, "vals");
  auto yIspace = runtime->create_index_space(ctx, Rect<1>(0, n - 1));
  auto xIspace = runtime->create_index_space(ctx, Rect<1>(0, m - 1));
  auto y = runtime->create_logical_region(ctx, yIspace, valFSpace);
  auto x = runtime->create_logical_region(ctx, xIspace, valFSpace);
  runtime->fill_field<double>(ctx, y, y, FID_VALUE, 0);
  {
    // TODO (rohany): Parallelize this operation.
    auto xreg = runtime->map_region(ctx, RegionRequirement(x, WRITE_DISCARD, EXCLUSIVE, x).add_field(FID_VALUE));
    FieldAccessor<WRITE_DISCARD,double,1,coord_t, Realm::AffineAccessor<double, 1, coord_t>> xrw(xreg, FID_VALUE);
    for (int i = 0; i < m; i++) {
      xrw[i] = i;
    }
    runtime->unmap_region(ctx, xreg);
  }

  auto A2_pos = A.indices[1][0];
  auto A2_pos_ispace = A2_pos.get_index_space();
  auto A2_crd = A.indices[1][1];
  auto A2_crd_ispace = A2_crd.get_index_space();
  auto A_vals = A.vals;
  auto A_vals_ispace = A_vals.get_index_space();

  auto domain = Domain(Rect<1>(0, pieces - 1));

  // Note: All of these partitioning operators need to use the parent region
  // which is the "region the task has privileges on" -- for the top level task
  // that is the region that they created, not the subregion.

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
    auto A_vals_logical_partition = runtime->get_logical_partition(ctx, A.valsParent, A_vals_partition);
    auto A2_crd_partition = runtime->create_index_partition(ctx, A2_crd_ispace, domain, A2_crd_col, LEGION_DISJOINT_COMPLETE_KIND);
    auto A2_crd_logical_partition = runtime->get_logical_partition(ctx, A.indicesParents[1][1], A2_crd_partition);

    // Now use the partition of A2_crd to partition A2_pos.
    auto A2_pos_part = runtime->create_partition_by_preimage_range(
      ctx,
      A2_crd_partition,
      A2_pos,
      A.indicesParents[1][0],
      FID_RECT_1,
      runtime->get_index_partition_color_space_name(A2_crd_partition)
    );
    auto A2_pos_logical_partition = runtime->get_logical_partition(ctx, A.indicesParents[1][0], A2_pos_part);

    // Use the partition bounds on A2_pos_logical_partition to partition y.
    for (PointInDomainIterator<1> itr(domain); itr(); itr++) {
      auto subreg = runtime->get_logical_subregion_by_color(ctx, A2_pos_logical_partition, DomainPoint(*itr));
      auto dom = runtime->get_index_space_domain(ctx, subreg.get_index_space());
      y_col[*itr] = Rect<1>(dom.lo(), dom.hi());
    }
    auto y_partition = runtime->create_index_partition(ctx, yIspace, domain, y_col, LEGION_ALIASED_COMPLETE_KIND);
    auto y_logical_partition = runtime->get_logical_partition(ctx, y, y_partition);

    RegionRequirement yReq = RegionRequirement(y_logical_partition, 0, LEGION_REDOP_SUM_FLOAT64, SIMULTANEOUS, y).add_field(FID_VALUE);
    RegionRequirement A2_pos_req = RegionRequirement(A2_pos_logical_partition, 0, READ_ONLY, EXCLUSIVE, A.indicesParents[1][0]).add_field(FID_RECT_1);
    RegionRequirement A2_crd_req = RegionRequirement(A2_crd_logical_partition, 0, READ_ONLY, EXCLUSIVE, A.indicesParents[1][1]).add_field(FID_INDEX);
    RegionRequirement A_vals_req = RegionRequirement(A_vals_logical_partition, 0, READ_ONLY, EXCLUSIVE, A.valsParent).add_field(FID_VALUE);
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
    auto A2_pos_logical_partition = runtime->get_logical_partition(ctx, A.indicesParents[1][0], A2_pos_partition);

    auto A2_crd_part = runtime->create_partition_by_image_range(
        ctx,
        A2_crd_ispace,
        A2_pos_logical_partition,
        A.indicesParents[1][0],
        FID_RECT_1,
        runtime->get_index_partition_color_space_name(ctx, A2_pos_partition)
    );
    auto A2_crd_logical_partition = runtime->get_logical_partition(ctx, A.indicesParents[1][1], A2_crd_part);

    // Use the crd partition to make a partition of the values.
    for (PointInDomainIterator<1> itr(domain); itr(); itr++) {
      auto subreg = runtime->get_logical_subregion_by_color(ctx, A2_crd_logical_partition, DomainPoint(*itr));
      auto subregIspace = subreg.get_index_space();
      auto bounds = runtime->get_index_space_domain(subregIspace);
      A_vals_col[*itr] = Rect<1>(bounds.lo(), bounds.hi());
    }
    auto A_vals_partition = runtime->create_index_partition(ctx, A_vals_ispace, domain, A_vals_col, LEGION_DISJOINT_COMPLETE_KIND);
    auto A_vals_logical_partition = runtime->get_logical_partition(ctx, A.valsParent, A_vals_partition);

    RegionRequirement yReq = RegionRequirement(yLogicalPartition, 0, READ_WRITE, EXCLUSIVE, y).add_field(FID_VALUE);
    RegionRequirement A2_pos_req = RegionRequirement(A2_pos_logical_partition, 0, READ_ONLY, EXCLUSIVE, A.indicesParents[1][0]).add_field(FID_RECT_1);
    RegionRequirement A2_crd_req = RegionRequirement(A2_crd_logical_partition, 0, READ_ONLY, EXCLUSIVE, A.indicesParents[1][1]).add_field(FID_INDEX);
    RegionRequirement A_vals_req = RegionRequirement(A_vals_logical_partition, 0, READ_ONLY, EXCLUSIVE, A.valsParent).add_field(FID_VALUE);
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

  if (dump) {
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

  for (int32_t i = in * ((n + (pieces - 1)) / pieces); i < in * ((n + (pieces - 1)) / pieces) + ((n + (pieces - 1)) / pieces); i++) {
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
    int32_t fposA = fpos0 * ((nnz + 3) / 4) + fpos1;
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
  {
    TaskVariantRegistrar registrar(TID_PRINT_COORD_LIST, "printCoordList");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<printCoordListMatrix>(registrar, "printCoordList");
  }
  {
    TaskVariantRegistrar registrar(TID_PACK_A_CSR, "packACSR");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<packACSR>(registrar, "packACSR");
  }
  return Runtime::start(argc, argv);
}