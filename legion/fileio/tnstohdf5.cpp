#include "legion.h"
#include "realm/cmdline.h"

#include "strings.h"
#include "hdf5_utils.h"

#include <hdf5.h>
#include <fstream>

using namespace Legion;

/*
 * This tool is used for converting .tns files into HDF5 files that can easily
 * be loaded into Legion through attach operations. This tool as two uses:
 *   ./bin/tnstohdf5 -tns f1 -hdf5 f2 // dump f1 into f2 as tns
 *   ./bin/tnstohdf5 -hdf5 f -dump // dump the contents of f
 * The tool dumps the tns into an HDF5 file with the following format:
 *              [coord0 coord1 coord2 ... coordN value]
 * where each coord0 is an int32_t, and value is a double. There are nnz
 * entries in the resulting data structure.
 */

enum TaskIDs {
  TID_TOP_LEVEL,
  TID_PRINT_DATA,
};

enum FieldIDs {
  FID_VALUE,
  FID_COORD, // This must be the last defined field.
};

static size_t numIntsToCompare = 0;
static int lexComp(const void* a, const void* b) {
  for (size_t i = 0; numIntsToCompare; i++) {
    int diff = ((int32_t*)a)[i] - ((int32_t*)b)[i];
    if (diff != 0) {
      return diff;
    }
  }
  return 0;
}

// This function performs a direct translation from a tns file into an HDF5 version
// of the tns file for easier interop with Legion later. It returns flat buffer that
// is an AOS representation of the coordinates and the value for each entry. The result
// buffer must be free'd by the user.
void* readTNSFile(std::string fileName, size_t& order, size_t& nnz) {
  // fileName must be a relative, sanitized path.
  std::fstream file;
  file.open(fileName, std::fstream::in);

  std::string line;
  if (!std::getline(file, line)) {
    std::cout << "Expected non-empty tns file." << std::endl;
    assert(false);
  }

  std::vector<std::string> toks = split(line, " ", false /* keepDelim */);
  order = toks.size() - 1;

  // The coordinates need to be sorted, otherwise we'll have to deal with the headache
  // of sorting them in legion. In order to do this (and keep the values along with the
  // coordinates) without doing a bunch of allocations, we have to drop into lower level code.
  // To do this, we'll allocate a flat buffer of "structs" where each struct has size
  // coords * coord_size + val_size. We can write individual elements into this buffer.
  // Then, we'll do a standard doubling array to grow the buffer as we read in elements.
  // This flat representation will let us then directly call qsort on the data like TACO.
  // If this becomes a bottleneck and we need to utilize a parallel sort, we can try and
  // implement the operation described here:
  // https://stackoverflow.com/questions/16874183/reusing-stdalgorithms-with-non-standard-containers/16905832

  // Set up the constants and buffers.
  size_t elemSize = order * sizeof(int32_t) + sizeof(double);
  size_t cnt = 0;
  // TODO (rohany): Adjust this initial size when loading larger files. It might
  //  even be a good idea to take this in via a command line parameter to have it
  //  fit to the correct size if we know it apriori.
  size_t size = 1;
  char* buffer = (char*)malloc(elemSize * size);

  // Load data from the tns file.
  do {
    // If we've hit the buffer capacity, then allocate a fresh buffer.
    if (cnt == size) {
      size *= 2;
      buffer = (char*)realloc(buffer, elemSize * size);
    }
    // Get the front of the buffer where we should insert into.
    char* insert = (elemSize * cnt) + buffer;
    char* linePtr = (char*)line.data();
    for (size_t i = 0; i < order; i++) {
      int32_t idx = strtol(linePtr, &linePtr, 10);
      assert(idx <= INT_MAX && "Coordinate in file is larger than INT_MAX");
      *(int32_t*)insert = idx;
      // Advance the pointer one int32_t position for the next coordinate.
      insert += sizeof(int32_t);
    }
    // After all of the int32_t coordinates, the last position remaining
    // is for the double.
    double val = strtod(linePtr, &linePtr);
    *(double*)insert = val;
    cnt++;
  } while (std::getline(file, line));

  // Clean up.
  file.close();

  // Now, let's sort the coordinates before dumping them. We'll use the qsort
  // function with a custom comparator.
  numIntsToCompare = order;
  qsort(buffer, cnt, elemSize, lexComp);

  nnz = cnt;
  return buffer;
}

void readHDF5Coords(const Task* task, const std::vector<PhysicalRegion> &regions, Context ctx, Runtime* runtime) {
  auto region = regions[0];

  // Declare accessors.
  typedef FieldAccessor<READ_ONLY,int32_t,1,coord_t, Realm::AffineAccessor<int32_t, 1, coord_t>> AccessorI;
  typedef FieldAccessor<READ_ONLY,double,1,coord_t, Realm::AffineAccessor<double, 1, coord_t>> AccessorD;

  std::vector<FieldID> fields;
  region.get_fields(fields);

  // Create an accessor for each of the fields corresponding to coordinates.
  std::vector<AccessorI> coords;
  for (size_t i = 0; i < fields.size() - 1; i++) {
    coords.push_back(AccessorI(region, FID_COORD + i));
  }

  // Create an accessor for the values.
  AccessorD vals(region, FID_VALUE);

  for (PointInRectIterator<1> itr(region); itr(); itr++) {
    for (auto acc : coords) {
      std::cout << acc[*itr] << " ";
    }
    std::cout << std::setprecision(9) << vals[*itr] << std::endl;
  }
}

std::vector<FieldID> fieldIDs(int order) {
  std::vector<FieldID> result;
  for (int i = 0; i < order; i++) {
    result.push_back(FID_COORD + i);
  }
  result.push_back(FID_VALUE);
  return result;
}

void allocateCoordFields(Context ctx, Runtime* runtime, std::vector<FieldID> fields, FieldSpace f) {
  Legion::FieldAllocator allocator = runtime->create_field_allocator(ctx, f);
  for (size_t i = 0; i < fields.size(); i++) {
    // Assume that the fields are laid out as [coord0, coord1, ..., coordN, value].
    if (i == fields.size() - 1) {
      allocator.allocate_field(sizeof(double), fields[i]);
    } else {
      allocator.allocate_field(sizeof(int32_t), fields[i]);
    }
  }
}

std::vector<std::string> constructFieldNames(std::vector<FieldID> fields) {
  std::vector<std::string> result;
  for (size_t i = 0; i < fields.size(); i++) {
    // Assume that the fields are laid out as [coord0, coord1, ..., coordN, value].
    if (i == fields.size() - 1) {
      result.push_back("value");
    } else {
      std::stringstream ss;
      ss << "coord" << i;
      result.push_back(ss.str());
    }
  }
  return result;
}

std::map<FieldID, const char*> constructFieldMap(std::vector<FieldID> fields, std::vector<std::string>& fieldNames) {
  std::map<FieldID, const char*> result;
  for (size_t i = 0; i < fields.size(); i++) {
    result[fields[i]] = fieldNames[i].c_str();
  }
  return result;
}

void top_level_task(const Task* task, const std::vector<PhysicalRegion> &regions, Context ctx, Runtime* runtime) {
  std::string tnsFilename;
  std::string hdf5Filename;
  bool dumpOnly = false;

  Realm::CommandLineParser parser;
  parser.add_option_string("-tns", tnsFilename);
  parser.add_option_string("-hdf5", hdf5Filename);
  parser.add_option_bool("-dump", dumpOnly);
  auto args = Runtime::get_input_args();
  assert(parser.parse_command_line(args.argc, args.argv));
  assert(!hdf5Filename.empty());

  if (dumpOnly) {
    // Look at the hdf5 file to get information about datasets and nnzs.
    auto result = getHDF5Info(hdf5Filename, "value");
    auto order = result.numFields - 1;
    auto nnz = result.numElements;

    // Create a region to represent the in-memory tns data.
    auto fspace = runtime->create_field_space(ctx);
    auto coordFieldIDs = fieldIDs(order);
    allocateCoordFields(ctx, runtime, coordFieldIDs, fspace);
    auto ispace = runtime->create_index_space(ctx, Rect<1>(0, nnz - 1));
    // Create regions for the in-memory and disk versions of the data.
    auto disk = runtime->create_logical_region(ctx, ispace, fspace); runtime->attach_name(disk, "disk");

    // Attach the region.
    auto fieldNames = constructFieldNames(coordFieldIDs);
    auto fieldMap = constructFieldMap(coordFieldIDs, fieldNames);
    AttachLauncher al(LEGION_EXTERNAL_HDF5_FILE, disk, disk);
    al.attach_hdf5(hdf5Filename.c_str(), fieldMap, LEGION_FILE_READ_WRITE);
    auto pdisk = runtime->attach_external_resource(ctx, al);

    // Launch a task to print out the result. This maps the region
    // into CPU memory for us to access directly.
    TaskLauncher launcher(TID_PRINT_DATA, TaskArgument());
    RegionRequirement req(disk, READ_ONLY, EXCLUSIVE, disk);
    for (size_t i = 0; i < order; i++) {
      req.add_field(coordFieldIDs[i]);
    }
    req.add_field(FID_VALUE);
    launcher.add_region_requirement(req);
    runtime->execute_task(ctx, launcher).wait();
    runtime->detach_external_resource(ctx, pdisk).wait();
    return;
  }

  // At this point, the tns filename must be defined.
  assert(!tnsFilename.empty());

  // Read in the .tns file into raw data in memory.
  size_t order, nnz;
  auto buf = readTNSFile(tnsFilename, order, nnz);

  // Create a region to represent the in-memory tns data.
  auto fspace = runtime->create_field_space(ctx);
  auto coordFieldIDs = fieldIDs(order);
  allocateCoordFields(ctx, runtime, coordFieldIDs, fspace);
  auto ispace = runtime->create_index_space(ctx, Rect<1>(0, nnz - 1));

  // Create regions for the in-memory and disk versions of the data.
  auto mem = runtime->create_logical_region(ctx, ispace, fspace); runtime->attach_name(mem, "in-memory");
  auto disk = runtime->create_logical_region(ctx, ispace, fspace); runtime->attach_name(disk, "disk");

  PhysicalRegion pmem, pdisk;
  // Get the local CPU memory.
  Memory sysmem = Machine::MemoryQuery(Machine::get_machine())
      .has_affinity_to(runtime->get_executing_processor(ctx))
      .only_kind(Memory::SYSTEM_MEM)
      .first();
  // Attach the in-memory data to the memory region.
  {
    AttachLauncher al(LEGION_EXTERNAL_INSTANCE, mem, mem);
    al.attach_array_aos(buf, false /* column_major */, coordFieldIDs, sysmem);
    pmem = runtime->attach_external_resource(ctx, al);
  }

  // Now, open up and attach the output hdf5 file.
  {
    // Create the field map. Allocate all of the field names up front
    // so that they are live until the actual attach call.
    auto fieldNames = constructFieldNames(coordFieldIDs);
    auto fieldMap = constructFieldMap(coordFieldIDs, fieldNames);
    std::vector<hid_t> hdf5Sizes;
    for (size_t i = 0; i < order; i++) {
      // Each of the coordinates is an int32_t.
      hdf5Sizes.push_back(H5T_NATIVE_INT32_g);
    }
    // The value is a double.
    hdf5Sizes.push_back(H5T_IEEE_F64LE_g);

    // Create the output hdf5 file.
    generateHDF5(hdf5Filename, fieldNames, hdf5Sizes, nnz);

    AttachLauncher al(LEGION_EXTERNAL_HDF5_FILE, disk, disk);
    al.attach_hdf5(hdf5Filename.c_str(), fieldMap, LEGION_FILE_READ_WRITE);
    pdisk = runtime->attach_external_resource(ctx, al);
  }

  // Finally, copy the in-memory instance into the disk instance.
  CopyLauncher cl;
  cl.add_copy_requirements(
      RegionRequirement(mem, READ_ONLY, EXCLUSIVE, mem),
      RegionRequirement(disk, WRITE_DISCARD, EXCLUSIVE, disk)
  );
  // Copy each of the coordinate fields and the value.
  for (size_t i = 0 ; i < order; i++) {
    cl.add_src_field(0, coordFieldIDs[i]);
    cl.add_dst_field(0, coordFieldIDs[i]);
  }
  cl.add_src_field(0, FID_VALUE);
  cl.add_dst_field(0, FID_VALUE);
  runtime->issue_copy_operation(ctx, cl);

  // Detach the external resources to flush any changes made.
  runtime->detach_external_resource(ctx, pmem).wait();
  runtime->detach_external_resource(ctx, pdisk).wait();

  // Free the buffer holding the data.
  free(buf);
}

int main(int argc, char** argv) {
  Runtime::set_top_level_task_id(TID_TOP_LEVEL);
  {
    TaskVariantRegistrar registrar(TID_TOP_LEVEL, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }
  {
    TaskVariantRegistrar registrar(TID_PRINT_DATA, "printData");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<readHDF5Coords>(registrar, "printData");
  }
  return Runtime::start(argc, argv);
}