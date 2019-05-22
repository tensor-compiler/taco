set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto -fvisibility=hidden -Wno-gnu-zero-variadic-macro-arguments")


add_subdirectory(pybind11)

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${TACO_INCLUDE_DIR})
include_directories(${TACO_PROJECT_DIR}/python_bindings/include)


pybind11_add_module(pytaco src/pytaco.cpp src/pyFormat.cpp src/pyDatatypes.cpp src/pyTensor.cpp
                    src/pyIndexNotation.cpp src/pyTensorIO.cpp)
set_target_properties(pytaco PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${TACO_PROJECT_DIR}/python_bindings/libs)
target_link_libraries(pytaco LINK_PRIVATE taco)

