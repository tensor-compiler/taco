find_package(Legion REQUIRED)

# Include the headers and supporting source files.
include_directories(include/ OpenBLAS/install/include "${CMAKE_BINARY_DIR}/include")
link_directories(OpenBLAS/install/lib)
file(GLOB LG_SOURCES src/*)

# Optimize these codes.
SET(CMAKE_CXX_FLAGS "-O2 ${CMAKE_CXX_FLAGS}")

function(add_app_folder folder)
    file(GLOB SOURCES "${folder}/main.cpp" "${folder}/taco-generated.cpp" ${LG_SOURCES})
    add_executable("${folder}" ${SOURCES})
    target_link_libraries("${folder}" Legion::Legion openblas)

    if (Legion_USE_CUDA)
        if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${folder}/taco-generated.cu")
            cuda_add_executable("${folder}-cuda" "${folder}/main.cpp" "${folder}/taco-generated.cu" ${LG_SOURCES})
            target_link_libraries("${folder}-cuda" Legion::Legion)
        endif()
    endif()
endfunction()

function(add_cuda_folder folder)
    if (Legion_USE_CUDA)
        file(GLOB SOURCES "${folder}/*.cpp" "${folder}/*.cu" ${LG_SOURCES})
        cuda_add_executable("${folder}" ${SOURCES})
        target_link_libraries("${folder}" Legion::Legion)
    endif()
endfunction()

add_app_folder(summaMM)
add_app_folder(cannonMM)
add_app_folder(placement-test)
add_app_folder(johnsonMM)
add_app_folder(solomonikMM)

add_cuda_folder(cuda-test)