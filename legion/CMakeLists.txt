find_package(Legion REQUIRED)

# Include the headers and supporting source files.
include_directories(include/ OpenBLAS/install/include)
link_directories(OpenBLAS/install/lib)
file(GLOB LG_SOURCES src/*)

# Optimize these codes.
SET(CMAKE_CXX_FLAGS "-O2 ${CMAKE_CXX_FLAGS}")

function(add_app_folder folder)
    file(GLOB SOURCES "${folder}/*.cpp" ${LG_SOURCES})
    add_executable("${folder}" ${SOURCES})
    target_link_libraries("${folder}" Legion::Legion openblas)
endfunction()

add_app_folder(summaMM)
add_app_folder(cannonMM)
add_app_folder(placement-test)
add_app_folder(johnsonMM)
add_app_folder(solomonikMM)
