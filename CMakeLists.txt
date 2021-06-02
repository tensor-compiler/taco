cmake_minimum_required(VERSION 3.4.0 FATAL_ERROR)
if(POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW)
endif()
if(POLICY CMP0054)
  cmake_policy(SET CMP0054 NEW)
endif()
project(taco
  VERSION 0.1
  LANGUAGES C CXX
)
option(CUDA "Build for NVIDIA GPU (CUDA must be preinstalled)" OFF)
option(PYTHON "Build TACO for python environment" OFF)
option(OPENMP "Build with OpenMP execution support" OFF)
option(COVERAGE "Build with code coverage analysis" OFF)
set(TACO_FEATURE_CUDA 0)
set(TACO_FEATURE_OPENMP 0)
set(TACO_FEATURE_PYTHON 0)
if(CUDA)
  message("-- Searching for CUDA Installation")
  find_package(CUDA REQUIRED)
  add_definitions(-DCUDA_BUILT)
  set(TACO_FEATURE_CUDA 1)
endif(CUDA)
if(OPENMP)
  message("-- Will use OpenMP for parallel execution")
  add_definitions(-DUSE_OPENMP)
  set(TACO_FEATURE_OPENMP 1)
endif(OPENMP)

if(PYTHON)
  message("-- Will build Python extension")
  add_definitions(-DPYTHON)
  set(TACO_FEATURE_PYTHON 1)
endif(PYTHON)

SET(CMAKE_CONFIGURATION_TYPES "Release;Debug;MinSizeRel;RelWithDebInfo")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING
    "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
    FORCE)
endif(NOT CMAKE_BUILD_TYPE)

if (CMAKE_BUILD_TYPE MATCHES Debug)
  message("-- Debug Build")
  add_definitions(-DTACO_DEBUG)
  add_definitions(-DTACO_ASSERTS)
  set(TACO_DEBUG 1)
elseif (CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
  message("-- Release Build with Debug Information")
  add_definitions(-DTACO_DEBUG)
  add_definitions(-DTACO_ASSERTS)
  set(TACO_DEBUG 1)
elseif (CMAKE_BUILD_TYPE MATCHES Release)
  message("-- Release Build")
elseif (CMAKE_BUILD_TYPE MATCHES MinSizeRel)
  message("-- Release Build with Minimal Size")
endif ()

if ($ENV{TACO_ASSERTS})
  add_definitions(-DTACO_ASSERTS)
endif ()

if (${CMAKE_SYSTEM_NAME} MATCHES "Windows")
  set(WINDOWS TRUE)
  add_definitions(-DTACO_WINDOWS)
elseif (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  set(LINUX TRUE)
  add_definitions(-DTACO_LINUX)
elseif (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(DARWIN TRUE)
  add_definitions(-DTACO_DARWIN)
  set(CMAKE_MACOSX_RPATH 1)
endif()

option(TACO_SHARED_LIBRARY "Build as a shared library" ON)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

set(OPTIMIZE "-O3" CACHE STRING "Optimization level")
set(C_CXX_FLAGS "-Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers -Wmissing-declarations -Woverloaded-virtual -pedantic-errors -Wno-deprecated")
if(OPENMP)
  set(C_CXX_FLAGS "-fopenmp ${C_CXX_FLAGS}")
endif(OPENMP)

if(COVERAGE)
  find_program(PATH_TO_GCOVR gcovr REQUIRED)
  # add coverage tooling to build flags
  set(C_CXX_FLAGS "${C_CXX_FLAGS} -g -fprofile-arcs -ftest-coverage")
  # name the coverage files "foo.gcno", not "foo.cpp.gcno"
  set(CMAKE_CXX_OUTPUT_EXTENSION_REPLACE 1)
  message("-- Code coverage analysis (gcovr) enabled")
endif(COVERAGE)

set(C_CXX_FLAGS "${C_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${C_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${C_CXX_FLAGS} -std=c++14")

set(TACO_PROJECT_DIR ${CMAKE_CURRENT_LIST_DIR})

set(TACO_SRC_DIR     ${TACO_PROJECT_DIR}/src)
set(TACO_TEST_DIR    ${TACO_PROJECT_DIR}/test)
set(TACO_TOOLS_DIR   ${TACO_PROJECT_DIR}/tools)
set(TACO_INCLUDE_DIR ${TACO_PROJECT_DIR}/include)

enable_testing()
include_directories(${TACO_INCLUDE_DIR})

set(TACO_LIBRARY_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

install(DIRECTORY ${TACO_INCLUDE_DIR}/ DESTINATION include FILES_MATCHING PATTERN "*.h")

add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(tools)
add_subdirectory(apps)
string(REPLACE " -Wmissing-declarations" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${TACO_PROJECT_DIR}/.git")
  # Update submodules as needed
  option(GIT_SUBMODULE "Check submodules during build" ON)
  if(GIT_SUBMODULE)
    message(STATUS "Submodule update")
    execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            RESULT_VARIABLE GIT_SUBMOD_RESULT)
    if(NOT GIT_SUBMOD_RESULT EQUAL "0")
      message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
    endif()
  endif()
  # get git revision
  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    RESULT_VARIABLE GIT_REVPARSE_RESULT
    OUTPUT_VARIABLE TACO_GIT_SHORTHASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(NOT GIT_REVPARSE_RESULT EQUAL "0")
    message(NOTICE "'git rev-parse --short HEAD' failed with ${GIT_REVPARSE_RESULT}, git version info will be unavailable.")
    set(TACO_GIT_SHORTHASH "")
  endif()
else()
  set(TACO_GIT_SHORTHASH "")
endif()

if(NOT EXISTS "${TACO_PROJECT_DIR}/python_bindings/pybind11/CMakeLists.txt")
  message(FATAL_ERROR "The submodules were not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()

if(PYTHON)
  add_subdirectory(python_bindings)
  message("-- Will build Python extension")
  add_definitions(-DPYTHON)
endif(PYTHON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wmissing-declarations")
add_custom_target(src DEPENDS apps)

if(COVERAGE)
  # code coverage analysis target
  add_custom_target(gcovr
    COMMAND mkdir -p coverage
    COMMAND ${CMAKE_MAKE_PROGRAM} test
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )
  add_custom_command(TARGET gcovr
    COMMAND echo "Running gcovr..."
    COMMAND ${PATH_TO_GCOVR} -r ${CMAKE_SOURCE_DIR} --html --html-details -o coverage/index.html ${CMAKE_BINARY_DIR}
    COMMAND echo "See coverage/index.html for coverage information."
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )
  add_dependencies(gcovr taco-test)
  if(PYTHON)
    add_dependencies(gcovr core_modules)
  endif(PYTHON)
  set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES coverage)
endif(COVERAGE)

string(TIMESTAMP TACO_BUILD_DATE "%Y-%m-%d")
configure_file("include/taco/version.h.in" "include/taco/version.h" @ONLY)
install(FILES "${CMAKE_BINARY_DIR}/include/taco/version.h" DESTINATION "include/taco")
