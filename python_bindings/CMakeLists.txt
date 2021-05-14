set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto -fvisibility=hidden -Wno-gnu-zero-variadic-macro-arguments")

add_subdirectory(pybind11)

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${TACO_INCLUDE_DIR})
include_directories(${TACO_PROJECT_DIR}/python_bindings/include)
if(CUDA)
  include_directories(${CUDA_INCLUDE_DIRS})
endif(CUDA)

set(PY_SRC_DIRS src)

foreach(dir ${PY_SRC_DIRS})
    file(GLOB PY_SOURCES ${PY_SOURCES} ${dir}/*.cpp)
endforeach()

file(GLOB PYTACO_FILES ${CMAKE_CURRENT_SOURCE_DIR}/pytaco/*.py)
file(GLOB PYTENSOR_FILES ${CMAKE_CURRENT_SOURCE_DIR}/pytaco/pytensor/*.py)

set(PY_SOURCES ${PY_SOURCES})
pybind11_add_module(core_modules ${PY_SOURCES} ${TACO_SOURCES})

set_target_properties(core_modules PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/pytaco/core)
target_link_libraries(core_modules LINK_PRIVATE taco)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/pytaco DESTINATION ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

set(PY_INSTALL_PATH ${CMAKE_INSTALL_PREFIX}/lib/pytaco)
install(FILES ${PYTACO_FILES} DESTINATION ${PY_INSTALL_PATH})
install(FILES ${PYTENSOR_FILES} DESTINATION ${PY_INSTALL_PATH}/pytensor)
install(TARGETS core_modules DESTINATION ${PY_INSTALL_PATH}/core)

configure_file("unit_tests.py" "unit_tests.py" @ONLY)
add_test(NAME pytaco COMMAND python3 unit_tests.py)
