#include "taco/spatial.h"

using namespace std;
namespace taco {

static bool Spatial_codegen_enabled = false;

bool should_use_Spatial_codegen() {
  return Spatial_codegen_enabled;
}

void set_Spatial_codegen_enabled(bool enabled) {
  Spatial_codegen_enabled = enabled;
}

}
