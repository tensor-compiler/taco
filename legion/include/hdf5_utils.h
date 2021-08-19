#include <hdf5.h>
#include <string>
#include <vector>

#include "legion.h"

// Create an HDF5 file with numElements elements, and the provided fields as individual datasets.
void generateHDF5(std::string fileName, std::vector<std::string> datasetNames, std::vector<hid_t> datasetSizes, int numElements);

// getHDF5Info gets information about an HDF5 file created by generateHDF5. To get the number of
// elements, it requires the user to pass in the name of a field that exists in the HDF5 file.
struct HDF5Info {
  size_t numFields;
  size_t numElements;
};
HDF5Info getHDF5Info(std::string fileName, std::string fieldName);

// Helper to attach a HDF5 file to a logical region. The returned PhysicalRegion
// must be explicitly deallocated with runtime->detach_external_resource.
Legion::PhysicalRegion attachHDF5(Legion::Context ctx, Legion::Runtime *runtime, Legion::LogicalRegion region,
                                  std::map<Legion::FieldID, const char *> fieldMap, std::string filename);
