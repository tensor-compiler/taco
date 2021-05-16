#include <Python.h>
#include <pybind11/pybind11.h>
#include "pyFormat.h"
#include "pyDatatypes.h"
#include "pyIndexNotation.h"
#include "pyTensor.h"
#include "pyTensorIO.h"
#include "pyParsers.h"


void addHelpers(py::module &m) {
  m.def("unique_name", (std::string(*)(char)) &taco::util::uniqueName);

  m.def("should_use_cuda_codegen", &taco::should_use_CUDA_codegen);

  py::options options;
  options.disable_function_signatures();

  m.def("get_num_threads", &taco::taco_get_num_threads, R"(
get_num_threads()

Get the number of threads taco uses for computation.

Taco defaults to using one thread to perform computations.

The number of threads can be increased using :func:`~set_num_threads` to perform computations in parallel. The
user is encouraged to tune the number of threads in order to increase performance.

Examples
---------
>>> import pytaco as pt
>>> pt.set_num_threads(1)
>>> pt.get_num_threads()
1

Returns
--------
number_of_threads: int
    The number of threads taco uses for performing tensor computations.
)");


  m.def("set_num_threads", &taco::taco_set_num_threads, py::arg("num_threads"), R"(
set_num_threads(num_threads)

Set the number of threads taco should use to perform computations.

The number of threads taco should use when running operations that can be done in parallel.

Parameters
------------
num_threads: int
    The number of threads taco should use when performing computations. The number of threads must be positive. Taco
    will ignore attempting to set the number of threads to a number less than or equal to 0.

Notes
------
Attempting to parallelize some expressions might lead to incorrect behaviour. In this case, taco will use one thread if
it determines an expression not parallelizable.

Examples
----------
>>> import pytaco as pt
>>> pt.set_num_threads(4) # tell taco to use 4 threads
>>> pt.get_num_threads()
4
>>> pt.set_num_threads(0) # ignored
>>> pt.get_num_threads() # Will be 4 since the last set was ignored
4
)");



  m.def("set_parallel_schedule", [](std::string sched_type, int chunk_size){
    std::transform(sched_type.begin(), sched_type.end(), sched_type.begin(), ::tolower);

    if(sched_type == "static") {
      taco::taco_set_parallel_schedule(taco::ParallelSchedule::Static, chunk_size);
    } else if (sched_type == "dynamic") {
      taco::taco_set_parallel_schedule(taco::ParallelSchedule::Dynamic, chunk_size);
    } else {
      py::value_error(R"(Schedule can only be "static" or "dynamic")");
    }
  }, R"(
set_parallel_schedule(sched_type, chunk_size)

Sets the strategy for performing computations in parallel.

Parameters
-----------
sched_type: string
    Either "static" or "dynamic". "static" indicates that Taco should parallelize 
    subsequent computations using a strategy that assigns the same number of 
    coordinates along a particular dimension to be processed by each thread. 
    "dynamic" indicates that Taco should parallelize subsequent computations 
    using a strategy that assigns work to the threads at runtime for better 
    load balance.

chunk_size: int
    For a dynamic schedule, the amount of additional work that is assigned to 
    any idle thread.

Notes
-------

Examples
---------



)", py::arg("sched_type"), py::arg("chunk_size") = 1);

  m.def("get_parallel_schedule", [](){
      taco::ParallelSchedule sched;
      int chunk_size = 0;
      taco::taco_get_parallel_schedule(&sched, &chunk_size);

      if(sched == taco::ParallelSchedule::Static) {
        return py::make_tuple("static", chunk_size);
      } else {
        return py::make_tuple("dynamic", chunk_size);
      }
  }, R"(
get_parallel_schedule()

Gets the current strategy for performing computations in parallel.

Examples
---------

Notes
-------

Returns
--------
schedule: tuple (string, int)
    A tuple where the first element indicates the strategy currently being used 
    to perform computations in parallel (either "static" or "dynamic") and the
    second element is the chunk size used for parallel computation.


)");


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

