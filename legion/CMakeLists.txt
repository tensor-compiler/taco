find_package(Legion REQUIRED)

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

# Optimize these codes.
SET(CMAKE_CXX_FLAGS "-O2 ${CMAKE_CXX_FLAGS}")

option(TACO_USE_LOGGING_MAPPER OFF)
if (TACO_USE_LOGGING_MAPPER)
    add_definitions(-DTACO_USE_LOGGING_MAPPER)
endif()

function(add_app_folder folder)
    file(GLOB SOURCES "${folder}/main.cpp" "${folder}/taco-generated.cpp" ${LG_SOURCES})
    add_executable("${folder}" ${SOURCES})
    target_link_libraries("${folder}" Legion::Legion openblas)

    if (Legion_USE_CUDA)
        if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${folder}/taco-generated.cu")
            cuda_add_executable("${folder}-cuda" "${folder}/main.cpp" "${folder}/taco-generated.cu" ${LG_SOURCES} ${LG_CU_SOURCES})
            target_link_libraries("${folder}-cuda" Legion::Legion cublas)
	    target_compile_definitions("${folder}-cuda" PUBLIC TACO_USE_CUDA)
        endif()
    endif()
endfunction()

function(add_cuda_folder folder)
    if (Legion_USE_CUDA)
        file(GLOB SOURCES "${folder}/*.cpp" "${folder}/*.cu" ${LG_SOURCES})
	cuda_add_executable("${folder}" ${SOURCES} ${LG_CU_SOURCES})
        target_link_libraries("${folder}" Legion::Legion)
	target_compile_definitions("${folder}" PUBLIC TACO_USE_CUDA)
    endif()
endfunction()

add_app_folder(summaMM)
add_app_folder(cannonMM)
add_app_folder(placement-test)
add_app_folder(johnsonMM)
add_app_folder(solomonikMM)

add_cuda_folder(cuda-test)
