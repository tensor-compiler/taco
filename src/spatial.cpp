#include "taco/spatial.h"

using namespace std;
namespace taco {

static bool Spatial_codegen_enabled = false;
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

}
