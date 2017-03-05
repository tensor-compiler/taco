cmake_minimum_required(VERSION 2.8)
project(sptv)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
file(GLOB SOURCE_CODE ${PROJECT_SOURCE_DIR}/*.cpp)
add_executable(${PROJECT_NAME} ${SOURCE_CODE})
set_target_properties("${PROJECT_NAME}" PROPERTIES OUTPUT_NAME "taco-${PROJECT_NAME}")

# Include taco headers and link to library
if (NOT DEFINED ENV{TACO_INCLUDE_DIR} OR NOT DEFINED ENV{TACO_LIBRARY_DIR})
  message(FATAL_ERROR "Set the environment variables TACO_INCLUDE_DIR and TACO_LIBRARY_DIR")
endif ()
include_directories($ENV{TACO_INCLUDE_DIR})
find_library(taco taco $ENV{TACO_LIBRARY_DIR})
target_link_libraries(${PROJECT_NAME} LINK_PUBLIC ${taco})
