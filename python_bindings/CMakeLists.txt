set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto -fvisibility=hidden -Wno-gnu-zero-variadic-macro-arguments")


add_subdirectory(pybind11)

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${TACO_INCLUDE_DIR})
include_directories(${TACO_PROJECT_DIR}/python_bindings/include)


pybind11_add_module(core_modules src/pytaco.cpp src/pyFormat.cpp src/pyDatatypes.cpp src/pyTensor.cpp
                    src/pyIndexNotation.cpp src/pyTensorIO.cpp src/pyParsers.cpp)

set_target_properties(core_modules PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${TACO_PROJECT_DIR}/python_bindings/pytaco/core)
target_link_libraries(core_modules LINK_PRIVATE taco)

