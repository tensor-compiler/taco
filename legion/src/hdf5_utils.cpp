#include "hdf5_utils.h"

using namespace Legion;

void generateCoordListHDF5(std::string filename, size_t order, size_t nnz) {
  // Open up the HDF5 file.
  hid_t fileID = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  assert(fileID >= 0);

  // Create a data space for the dimensions.
  hsize_t dims[1];
  dims[0] = order;
  hid_t dimensionsDataspaceID = H5Screate_simple(1, dims, NULL);
  assert(dimensionsDataspaceID >= 0);
  // Create a data space for the coordinates and values.
  dims[0] = nnz;
  hid_t coordSpaceID = H5Screate_simple(1, dims, NULL);
  assert(coordSpaceID >= 0);

  std::vector<hid_t> datasets;
  auto createDataset = [&](std::string name, hid_t size, hid_t dataspaceID) {
    hid_t dataset = H5Dcreate2(fileID, name.c_str(), size, dataspaceID, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    assert(dataset >= 0);
    datasets.push_back(dataset);
  };

  // Create a dataset for each of the coordinates.
  for (size_t i = 0; i < order; i++) {
    // We'll use int32_t's for the coordinates.
    createDataset(COOCoordsFields[i], H5T_NATIVE_INT32_g, coordSpaceID);
  }
  // Create a dataset for the values.
  createDataset(COOValsField, H5T_IEEE_F64LE_g, coordSpaceID);
  // Create a dataset for the dimensions.
  createDataset(COODimsField, H5T_NATIVE_INT32_g, dimensionsDataspaceID);

  // Close up everything now.
  for (auto id : datasets) {
    H5Dclose(id);
  }
  H5Sclose(coordSpaceID);
  H5Sclose(dimensionsDataspaceID);
  H5Fclose(fileID);
}

void getCoordListHDF5Meta(std::string filename, size_t& order, size_t& nnz) {
  auto hdf5file = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  auto getDataSetDim = [&](const char* dataSetName) {
    auto dataset = H5Dopen1(hdf5file, dataSetName);
    auto dataSpace = H5Dget_space(dataset);
    hsize_t dims[1];
    H5Sget_simple_extent_dims(dataSpace, dims, NULL);
    auto result = size_t(dims[0]);
    H5Sclose(dataSpace);
    H5Dclose(dataset);
    return result;
  };
  order = getDataSetDim(COODimsField);
  nnz = getDataSetDim(COOValsField);
  H5Fclose(hdf5file);
}

PhysicalRegion attachHDF5(Context ctx, Runtime *runtime, LogicalRegion region, std::map<FieldID, const char *> fieldMap, std::string filename, Legion::LegionFileMode mode) {
  AttachLauncher al(LEGION_EXTERNAL_HDF5_FILE, region, region);
  al.attach_hdf5(filename.c_str(), fieldMap, mode);
  return runtime->attach_external_resource(ctx, al);
}
