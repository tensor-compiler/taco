#include "taco/spatial.h"

using namespace std;
namespace taco {

static bool Spatial_codegen_enabled = false;
static bool Spatial_multi_dim_enabled = false;
static bool output_store = true;
static bool tensor_files = false;

static int Spatial_default_dimension = 0;

bool should_use_Spatial_codegen() {
  return Spatial_codegen_enabled;
}

void set_Spatial_codegen_enabled(bool enabled) {
  Spatial_codegen_enabled = enabled;
}

void set_Spatial_dimension(int dim) {
  Spatial_default_dimension = dim;
}

int get_Spatial_dimension() {
  return Spatial_default_dimension;
}

bool should_use_Spatial_multi_dim() {
  return Spatial_multi_dim_enabled;
}

void set_output_store(bool enabled) {
  output_store = enabled;
}

bool should_output_store() {
  return output_store;
}

void set_tensor_files(bool enabled) {
  tensor_files = enabled;
}

bool should_use_tensor_files() {
  return tensor_files;
}

}