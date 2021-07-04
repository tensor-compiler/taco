cmake_minimum_required(VERSION 3.11)
if(POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW)
endif()
project(tensor_times_vector)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
file(GLOB SOURCE_CODE ${PROJECT_SOURCE_DIR}/*.cpp)
add_executable(${PROJECT_NAME} ${SOURCE_CODE})

# To let the app be a standalone project 
# satisfy taco dep from source if not found on system
if(NOT TARGET taco::taco)
  include(FetchContent)

  # To prevent download of the pybind11 source you can overwrite the following
  # behavior by setting the cmake variable FETCHCONTENT_SOURCE_DIR_PYBIND11 to a
  # local source directory
  FetchContent_Declare(
    taco
    GIT_REPOSITORY https://github.com/tensor-compiler/taco.git
    GIT_TAG        master
  )
  FetchContent_MakeAvailable(taco)
endif()

target_link_libraries(${PROJECT_NAME} LINK_PUBLIC taco::taco)

