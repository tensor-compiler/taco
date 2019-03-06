set(PYBIND11_CPP_STANDARD -std=c++11)

find_package(pybind11 REQUIRED PATHS ~/pybind11)

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${TACO_INCLUDE_DIR})
include_directories(${TACO_PROJECT_DIR}/python_bindings/include)

pybind11_add_module(pytaco src/pytaco.cpp)
set_target_properties(pytaco PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${TACO_PROJECT_DIR}/python_bindings/libs)
target_link_libraries(pytaco PRIVATE ${PYTHON_LIBRARIES})
target_link_libraries(pytaco LINK_PRIVATE taco)

