#include <Python.h>
#include <pybind11/pybind11.h>
#include "pyFormat.h"
#include "pyDatatypes.h"
#include "pyIndexNotation.h"
#include "pyTensor.h"
#include "pyTensorIO.h"
#include "pyParsers.h"


void addHelpers(py::module &m) {
  m.def("taco_get_num_threads", &taco::taco_get_num_threads);
  m.def("taco_set_num_threads", &taco::taco_set_num_threads, py::arg("num_threads"));
  m.def("unique_name", (std::string(*)(char)) &taco::util::uniqueName);

  m.def("taco_set_parallel_schedule", [](std::string sched_type, int chunk_size = 0){
    std::transform(sched_type.begin(), sched_type.end(), sched_type.begin(), ::tolower);

    if(sched_type == "static") {
      taco::taco_set_parallel_schedule(taco::ParallelSchedule::Static, chunk_size);
    } else if (sched_type == "dynamic") {
      taco::taco_set_parallel_schedule(taco::ParallelSchedule::Dynamic, chunk_size);
    } else {
      py::value_error(R"(Schedule can only be "static" or "dynamic")");
    }
  });

  m.def("taco_get_parallel_schedule", [](){
      taco::ParallelSchedule sched;
      int chunk_size = 0;
      taco::taco_get_parallel_schedule(&sched, &chunk_size);

      if(sched == taco::ParallelSchedule::Static) {
        return py::make_tuple("static", chunk_size);
      } else {
        return py::make_tuple("dynamic", chunk_size);
      }
  });


}

PYBIND11_MODULE(core_modules, m){

  m.doc() = "A Python module for operating on Sparse Tensors.";
  using namespace taco::pythonBindings;
  addHelpers(m);
  defineTacoTypes(m);
  defineModeFormats(m);
  defineModeFormatPack(m);
  defineFormat(m);
  defineIndexNotation(m);
  defineTensor(m);
  defineIOFuncs(m);
  defineParser(m);

}

