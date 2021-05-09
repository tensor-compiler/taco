#include "legion.h"
#include <iostream>

using namespace Legion;

const int TID_TOP_LEVEL = 1;

void top_level_task(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  std::cout << "Hello from legion!" << std::endl;
}

int main(int argc, char **argv) {
  Runtime::set_top_level_task_id(TID_TOP_LEVEL);
  {
    TaskVariantRegistrar registrar(TID_TOP_LEVEL, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }
  return Runtime::start(argc, argv);
}
