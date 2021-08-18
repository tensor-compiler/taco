#include "hdf5_utils.h"

void generateHDF5(std::string fileName, std::vector<std::string> datasetNames, std::vector<hid_t> datasetSizes, int numElements) {
  hid_t file_id = H5Fcreate(fileName.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  assert(file_id >= 0);

  hsize_t dims[1];
  dims[0] = numElements;
  hid_t dataspace_id = H5Screate_simple(1, dims, NULL);
  assert(dataspace_id >= 0);

  // Create a new dataset for each field?
  std::vector<hid_t> datasets;
  for (size_t i = 0; i < datasetNames.size(); i++) {
    hid_t dataset = H5Dcreate2(file_id, datasetNames[i].c_str(), datasetSizes[i], dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    assert(dataset >= 0);
    datasets.push_back(dataset);
  }

  // Close things up for attach open up later.
  for (auto dataset : datasets) {
    H5Dclose(dataset);
  }
  H5Sclose(dataspace_id);
  H5Fclose(file_id);
}

HDF5Info getHDF5Info(std::string fileName, std::string fieldName) {
  // TODO (rohany): This is kind of hacky, as we are assuming that the datasets in the
  //  hdf5 file are flat, so that the total number is actually just the number of columns.
  HDF5Info result;
  auto hdf5file = H5Fopen(fileName.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  H5G_info_t info;
  assert(H5Gget_info(hdf5file, &info) >= 0);
  result.numFields = info.nlinks;
  auto dset = H5Dopen1(hdf5file, fieldName.c_str());
  assert(dset >= 0);
  auto dataSpace = H5Dget_space(dset);
  assert(dataSpace >= 0);
  // This is a single-dimension dataspace.
  hsize_t dims[1], maxDims[1];
  H5Sget_simple_extent_dims(dataSpace, dims, maxDims);
  assert(dims[0] == maxDims[0]);
  result.numElements = dims[0];
  // Close the opened resources.
  H5Sclose(dataSpace);
  H5Dclose(dset);
  H5Fclose(hdf5file);
  return result;
}
