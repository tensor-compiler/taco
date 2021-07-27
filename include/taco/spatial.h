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

/// Check if should use Spatial multi-dimensional memories
bool should_use_Spatial_multi_dim();

void set_output_store(bool enabled);

bool should_output_store();

void set_tensor_files(bool enabled);
bool should_use_tensor_files();
}

#endif

