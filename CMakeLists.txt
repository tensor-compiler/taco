cmake_minimum_required(VERSION 2.8.3 FATAL_ERROR)
project(taco)

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
set(C_CXX_FLAGS "${C_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${C_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${C_CXX_FLAGS} -std=c++11")

set(TACO_PROJECT_DIR ${CMAKE_CURRENT_LIST_DIR})

set(TACO_SRC_DIR     ${TACO_PROJECT_DIR}/src)
set(TACO_TEST_DIR    ${TACO_PROJECT_DIR}/test)
set(TACO_TOOLS_DIR   ${TACO_PROJECT_DIR}/tools)
set(TACO_INCLUDE_DIR ${TACO_PROJECT_DIR}/include)

include_directories(${TACO_INCLUDE_DIR})

set(TACO_LIBRARY_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

install(DIRECTORY ${TACO_INCLUDE_DIR}/ DESTINATION include)

add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(tools)
add_subdirectory(apps)
add_custom_target(src DEPENDS apps)
