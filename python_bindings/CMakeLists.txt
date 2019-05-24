set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto -fvisibility=hidden -Wno-gnu-zero-variadic-macro-arguments")

add_subdirectory(pybind11)

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${TACO_INCLUDE_DIR})
include_directories(${TACO_PROJECT_DIR}/python_bindings/include)

set(PY_SRC_DIRS src)

foreach(dir ${PY_SRC_DIRS})
    file(GLOB PY_SOURCES ${PY_SOURCES} ${dir}/*.cpp)
endforeach()

set(PY_SOURCES ${PY_SOURCES})
pybind11_add_module(core_modules ${PY_SOURCES} ${TACO_SOURCES})

set_target_properties(core_modules PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${TACO_PROJECT_DIR}/python_bindings/pytaco/core)
target_link_libraries(core_modules LINK_PRIVATE taco)

