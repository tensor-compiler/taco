#ifndef SPATIAL_H
#define SPATIAL_H

#include <string>
#include <sstream>
#include <ostream>

namespace taco {
/// Functions used by taco to interface with Spatial
/// Check if should use Spatial codegen
bool should_use_Spatial_codegen();
/// Enable/Disable Spatial codegen
void set_Spatial_codegen_enabled(bool enabled);
void set_Spatial_dimension(int dim); 
int get_Spatial_dimension();

}

#endif

