#include <python3.6/Python.h>
#include "pyeinsum.h"
#include "pytensor.h"

PYBIND11_MODULE(pytaco, m){

  m.doc() = "A Python module for operating on Sparse Tensors.";

}