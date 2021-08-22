find_package(Legion REQUIRED)
if(OPENMP)
  find_package(OpenMP REQUIRED)
endif(OPENMP)

# We need to have HDF5_ROOT set here to do this if we're using
# a non-system provided HDF5 installation.
find_package(HDF5 REQUIRED)

# Include the headers and supporting source files.
include_directories(include/ OpenBLAS/install/include "${CMAKE_BINARY_DIR}/include")
link_directories(OpenBLAS/install/lib)
file(GLOB LG_SOURCES src/*.cpp)
file(GLOB LG_CU_SOURCES src/*.cu)
if (Legion_USE_CUDA)
  if (NOT DEFINED TACO_CUDA_LIBS)
    message(ERROR "Please set TACO_CUDA_LIBS to the path to CUDA libraries")
  endif()
  link_directories(${TACO_CUDA_LIBS})
endif()

# Collect sources that can be also used by tools outside of TACO codes.
set(LG_EXTRA_SOURCES src/hdf5_utils.cpp src/strings.cpp)

# Optimize these codes.
SET(CMAKE_CXX_FLAGS "-O3 ${CMAKE_CXX_FLAGS}")

option(TACO_USE_LOGGING_MAPPER OFF)
if (TACO_USE_LOGGING_MAPPER)
    add_definitions(-DTACO_USE_LOGGING_MAPPER)
endif()


# We're targeting GPUs with at least -sm_60 capability. We need this so that
# we can use the atomicAdd() function in leaf kernels.
# On Lassen, we need to target -sm_70.
if (DEFINED ENV{LLNL_COMPUTE_NODES})
  set(CUDA_NVCC_FLAGS -gencode arch=compute_70,code=sm_70)
else()
  set(CUDA_NVCC_FLAGS -gencode arch=compute_60,code=sm_60)
endif()

function(add_app_folder folder)
    file(GLOB SOURCES "${folder}/main.cpp" "${folder}/taco-generated.cpp" ${LG_SOURCES})
    add_executable("${folder}" ${SOURCES})
    target_link_libraries("${folder}" Legion::Legion openblas)
    if (OPENMP)
      target_compile_options("${folder}" PRIVATE ${OpenMP_CXX_FLAGS})
    endif(OPENMP)

    if (Legion_USE_CUDA)
        if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${folder}/taco-generated.cu")
            cuda_add_executable("${folder}-cuda" "${folder}/main.cpp" "${folder}/taco-generated.cu" ${LG_SOURCES} ${LG_CU_SOURCES})
            target_link_libraries("${folder}-cuda" Legion::Legion cublas -fopenmp)
	    # set_property(TARGET "${folder}-cuda" PROPERTY CUDA_ARCHITECTURES 60)
	    # target_compile_options("${folder}-cuda" PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_60>)
	    # target_compile_options("${folder}-cuda" PUBLIC -gencode arch=compute_50,code=sm_50)
            if (OPENMP)
                target_link_libraries("${folder}-cuda" -fopenmp)
                target_compile_options("${folder}-cuda" PUBLIC -fopenmp)
            endif(OPENMP)
            target_compile_definitions("${folder}-cuda" PUBLIC TACO_USE_CUDA)
            set_property(TARGET "${folder}-cuda" PROPERTY CUDA_STANDARD 14)
        endif()
    endif()
endfunction()

function(add_cuda_folder folder)
    if (Legion_USE_CUDA)
        file(GLOB SOURCES "${folder}/*.cpp" "${folder}/*.cu" ${LG_SOURCES})
        cuda_add_executable("${folder}" ${SOURCES} ${LG_CU_SOURCES})
        target_link_libraries("${folder}" Legion::Legion)
        target_compile_definitions("${folder}" PUBLIC TACO_USE_CUDA)
        set_property(TARGET "${folder}" PROPERTY CUDA_STANDARD 14)
    endif()
endfunction()

set(TEST_BINARIES "")
macro(add_basic_legion_test binary tag)
    # TODO (rohany): See what more runtime debug flags I can place here.
    if (OPENMP)
        add_test(NAME "legion-${binary}-${tag}" COMMAND ${binary} -lg:partcheck -lg:safe_mapper -ll:ocpu 1 -ll:othr 1 -ll:csize 5000 ${ARGN})
    else()
        add_test(NAME "legion-${binary}-${tag}" COMMAND ${binary} -lg:partcheck -lg:safe_mapper -ll:cpu 1 -ll:csize 5000 ${ARGN})
    endif()
    # Collect all the test binaries that we depend on.
    list(APPEND TEST_BINARIES "${binary}")
endmacro()
macro(add_basic_legion_gpu_test binary tag)
    if (Legion_USE_CUDA)
        add_test(NAME "legion-${binary}-cuda-${tag}" COMMAND "${binary}-cuda" -lg:partcheck -lg:safe_mapper -ll:ocpu 1 -ll:othr 1 -ll:gpu 1 -ll:csize 5000 -ll:fsize 2048 ${ARGN})
        # Collect all the test binaries that we depend on.
        list(APPEND TEST_BINARIES "${binary}-cuda")
    endif()
endmacro()
macro(add_basic_legion_cpugpu_test binary tag)
    add_basic_legion_test(${binary} ${tag} ${ARGN})
    add_basic_legion_gpu_test(${binary} ${tag} ${ARGN})
endmacro()

add_app_folder(pummaMM)
add_app_folder(summaMM)
add_app_folder(cannonMM)
add_app_folder(placement-test)
add_app_folder(johnsonMM)
add_app_folder(solomonikMM)
add_app_folder(cosma)
add_app_folder(ttv)
add_app_folder(mttkrp)
add_app_folder(ttmc)
add_app_folder(innerprod)
add_app_folder(matvec-adapt)

add_cuda_folder(cuda-test)

# Add a variety of tests for the generated code.

# Tests for cannon.
add_basic_legion_cpugpu_test(cannonMM basic -gx 4 -gy 4 -n 2048)
add_basic_legion_cpugpu_test(cannonMM nondiv1 -gx 4 -gy 4 -n 2047)
add_basic_legion_cpugpu_test(cannonMM nondiv2 -gx 4 -gy 4 -n 2049)

# Tests for SUMMA.
add_basic_legion_test(summaMM basic -gx 4 -gy 4 -n 2048)
add_basic_legion_test(summaMM nondiv1 -gx 4 -gy 4 -n 2047)
add_basic_legion_test(summaMM nondiv2 -gx 4 -gy 4 -n 2049)

# Tests for Johnson.
add_basic_legion_test(johnsonMM basic -n 1024 -gdim 3)

# Tests for Solomonik-2.5D.
add_basic_legion_test(solomonikMM basic -n 1024)

# Tests for the matvec-adapt stuff. Run with and without a match.
add_basic_legion_test(matvec-adapt match -n 1024 -pieces 8 -match)
add_basic_legion_test(matvec-adapt nomatch -n 1024 -pieces 8)

# Deduplicate the list of test binaries so that we have all of our dependencies.
# All tests must be added above this line.
list(REMOVE_DUPLICATES TEST_BINARIES)
add_custom_target(test-legion COMMAND ctest -R legion DEPENDS ${TEST_BINARIES})

# Add some hand written codes here.
add_executable(spmv SpMV/handwritten.cpp ${LG_EXTRA_SOURCES})
include_directories(spmv PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(spmv Legion::Legion ${HDF5_LIBRARIES})
target_compile_options(spmv PRIVATE ${OpenMP_CXX_FLAGS})

# Utility to preload tns files into hdf5 files for easier ingestion.
add_executable(tensor_to_hdf5 fileio/tensor_to_hdf5.cpp ${LG_EXTRA_SOURCES})
target_link_libraries(tensor_to_hdf5 Legion::Legion ${HDF5_LIBRARIES})
include_directories(tensor_to_hdf5 PRIVATE ${HDF5_INCLUDE_DIRS})
