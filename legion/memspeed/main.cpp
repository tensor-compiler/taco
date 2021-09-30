#include "legion.h"
#include "mappers/default_mapper.h"

using namespace Legion;
using namespace Legion::Mapping;

enum TaskIDs {
  TID_TOP_LEVEL,
  TID_DUMMY_READ,
};
enum FieldIDs {
  FID_VAL,
};

class BenchMapper : public DefaultMapper {
public:
  BenchMapper(MapperRuntime* rt, Machine& machine, const Legion::Processor& local) : DefaultMapper(rt, machine, local) {}

  void slice_task(const MapperContext ctx,
                  const Task &task,
                  const SliceTaskInput &input,
                  SliceTaskOutput &output) {
    // Depending on the requested node, we'll slice onto a different set of processors.
    auto addressSpace = *(AddressSpace*)(task.args);
    Machine::ProcessorQuery gpuQuery(this->machine);
    gpuQuery.only_kind(Processor::TOC_PROC);
    gpuQuery.same_address_space_as(this->remote_gpus[addressSpace]);
    auto procs = std::vector<Legion::Processor>{gpuQuery.begin(), gpuQuery.end()};
    switch (input.domain.get_dim())
    {
#define BLOCK(DIM) \
        case DIM: \
          { \
            DomainT<DIM,coord_t> point_space = input.domain; \
            Point<DIM,coord_t> num_blocks = \
              default_select_num_blocks<DIM>(procs.size(), point_space.bounds); \
            default_decompose_points<DIM>(point_space, procs, \
                  num_blocks, false/*recurse*/, \
                  stealing_enabled, output.slices); \
            break; \
          }
      LEGION_FOREACH_N(BLOCK)
#undef BLOCK
      default: // don't support other dimensions right now
        assert(false);
    }
  }
};

void register_mapper(Machine m, Runtime* runtime, const std::set<Processor>& local_procs) {
  runtime->replace_default_mapper(new BenchMapper(runtime->get_mapper_runtime(), m, *local_procs.begin()), Processor::NO_PROC);
}

// dummy does nothing, as we want to see if we can saturate the bandwidth between two nodes.
void dummy(const Task* task, const std::vector<PhysicalRegion>&, Context ctx, Runtime* runtime) {}

void top_level_task(const Task* task, const std::vector<PhysicalRegion>&, Context ctx, Runtime* runtime) {
  // Set up some constants.
  AddressSpace node0 = 0;
  AddressSpace node1 = 1;
  // Create some data. We'll set this up so that each GPU has a 1G chunk that they will send
  // to each other GPU in the system. I'm assuming the Lassen node architecture, so we'll have
  // 4G of data for 4 GPUs.
  auto size = 500000000;
  auto fspace = runtime->create_field_space(ctx);
  {
    Legion::FieldAllocator alloc = runtime->create_field_allocator(ctx, fspace);
    alloc.allocate_field(sizeof(double), FID_VAL);
  }
  // TODO (rohany): Depending on the performance, we can change this stuff from 1D to 2D copies etc.
  auto ispace = runtime->create_index_space(ctx, Rect<1>{0, size * 4});
  auto cspace = runtime->create_index_space(ctx, Rect<1>(0, 3));
  auto reg = runtime->create_logical_region(ctx, ispace, fspace);
  auto part = runtime->create_equal_partition(ctx, ispace, cspace);
  auto lpart = runtime->get_logical_partition(ctx, reg, part);
  // Launch a dummy onto node 1 to create the valid instances.
  {
    IndexLauncher launcher(TID_DUMMY_READ, runtime->get_index_space_domain(cspace), TaskArgument(&node0, sizeof(AddressSpace)), ArgumentMap());
    launcher.add_region_requirement(RegionRequirement(lpart, 0, WRITE_DISCARD, EXCLUSIVE, reg).add_field(FID_VAL));
    runtime->execute_index_space(ctx, launcher).wait_all_results();
  }
  // Now, launch a task onto each GPU on the other node.
  auto start = runtime->get_current_time(ctx);
  {
    IndexLauncher launcher(TID_DUMMY_READ, runtime->get_index_space_domain(cspace), TaskArgument(&node1, sizeof(AddressSpace)), ArgumentMap());
    launcher.add_region_requirement(RegionRequirement(lpart, 0, READ_ONLY, EXCLUSIVE, reg).add_field(FID_VAL));
    runtime->execute_index_space(ctx, launcher);
  }
  runtime->issue_execution_fence(ctx);
  auto end = runtime->get_current_time(ctx);
  auto s = double((end.get<double>() - start.get<double>()));
  auto totalDataSizeGB = double(size * 4 * sizeof(double)) / 1e9;
  auto gbs = totalDataSizeGB / s;
  std::cout << "Achieved aggregate GB/s of " << gbs << "." << std::endl;
}

int main(int argc, char** argv) {
  Runtime::set_top_level_task_id(TID_TOP_LEVEL);
  {
    TaskVariantRegistrar registrar(TID_TOP_LEVEL, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_replicable(false);
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }
  {
    TaskVariantRegistrar registrar(TID_DUMMY_READ, "dummy");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    Runtime::preregister_task_variant<dummy>(registrar, "dummy");
  }
  Runtime::add_registration_callback(register_mapper);
  return Runtime::start(argc, argv);
}
