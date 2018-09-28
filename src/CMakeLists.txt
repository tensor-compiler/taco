if (TACO_SHARED_LIBRARY)
  set(TACO_LIBRARY_TYPE SHARED)
  message("-- Shared library")
else()
  set(TACO_LIBRARY_TYPE STATIC)
  message("-- Static library")
endif()

find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

# Find the libraries that correspond to the LLVM components
# that we wish to use
llvm_map_components_to_libnames(llvm_libs core mcjit bitwriter linker X86 passes)

set(TACO_SRC_DIRS . parser index_notation lower ir codegen storage error util)

foreach(dir ${TACO_SRC_DIRS})
  file(GLOB TACO_HEADERS ${TACO_HEADERS} ${dir}/*.h)
  file(GLOB TACO_HEADERS ${TACO_HEADERS} ${TACO_INCLUDE_DIR}/taco/${dir}/*.h)
  file(GLOB TACO_SOURCES ${TACO_SOURCES} ${dir}/*.cpp)
endforeach()

set(TACO_HEADERS ${TACO_HEADERS})
set(TACO_SOURCES ${TACO_SOURCES})

add_definitions(${TACO_DEFINITIONS})
include_directories(${TACO_SRC_DIR})
add_library(taco ${TACO_LIBRARY_TYPE} ${TACO_HEADERS} ${TACO_SOURCES})
install(TARGETS taco DESTINATION lib)

if (LINUX)
  target_link_libraries(taco PRIVATE ${TACO_LIBRARIES} dl ${llvm_libs})
else()
  target_link_libraries(taco PRIVATE ${TACO_LIBRARIES} ${llvm_libs})
endif()

