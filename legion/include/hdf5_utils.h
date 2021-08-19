#ifndef TACO_LEGION_HDF5_UTILS_H
#define TACO_LEGION_HDF5_UTILS_H
#include <hdf5.h>
#include <string>
#include <vector>

#include "legion.h"

// Constants for coordinate list HDF5 storage.
const char* const COODimsField = "dims";
const char* const COOValsField = "vals";
const char* const COOCoordsFields[] = {
    "coord0",
    "coord1",
    "coord2",
    "coord3",
    "coord4",
    "coord5",
    "coord6",
    "coord7",
    // TODO (rohany): Add more if needed.
};
// generateCoordListHDF5 creates a serialized coordinate list representation of a
// tensor in an HDF5 file. It uses int32_t's to store coordinates and dimension
// information and double's to store the values. The generated HDF5 file has
// the following structure:
//   dims: ...
//   vals: ...
//   coord0: ...
//   coord1: ...
//   ...
void generateCoordListHDF5(std::string filename, size_t order, size_t nnz);
// getCoordListHDF5Meta extracts metadata from an HDF5 file created by generateCoordListHDF5.
void getCoordListHDF5Meta(std::string filename, size_t& order, size_t& nnz);

// Helper to attach a HDF5 file to a logical region. The returned PhysicalRegion
// must be explicitly deallocated with runtime->detach_external_resource.
Legion::PhysicalRegion attachHDF5(Legion::Context ctx, Legion::Runtime *runtime, Legion::LogicalRegion region,
                                  std::map<Legion::FieldID, const char *> fieldMap, std::string filename,
                                  Legion::LegionFileMode mode = LEGION_FILE_READ_ONLY);

#endif // TACO_LEGION_HDF5_UTILS_H