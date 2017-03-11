cmake_minimum_required(VERSION 2.8)
project(tensor_times_vector)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
file(GLOB SOURCE_CODE ${PROJECT_SOURCE_DIR}/*.cpp)
add_executable(${PROJECT_NAME} ${SOURCE_CODE})

# To let the app be a standalone project 
if (NOT TACO_INCLUDE_DIR)
  if (NOT DEFINED ENV{TACO_INCLUDE_DIR} OR NOT DEFINED ENV{TACO_LIBRARY_DIR})
    message(FATAL_ERROR "Set the environment variables TACO_INCLUDE_DIR and TACO_LIBRARY_DIR")
  endif ()
  set(TACO_INCLUDE_DIR $ENV{TACO_INCLUDE_DIR})
  set(TACO_LIBRARY_DIR $ENV{TACO_LIBRARY_DIR})
  find_library(taco taco ${TACO_LIBRARY_DIR})
  target_link_libraries(${PROJECT_NAME} LINK_PUBLIC ${taco})
else()
  set_target_properties("${PROJECT_NAME}" PROPERTIES OUTPUT_NAME "taco-${PROJECT_NAME}")
  target_link_libraries(${PROJECT_NAME} LINK_PUBLIC taco)
endif ()

# Include taco headers
include_directories(${TACO_INCLUDE_DIR})
