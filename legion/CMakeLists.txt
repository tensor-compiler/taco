find_package(Legion REQUIRED)

# Include the headers and supporting source files.
include_directories(include/)
file(GLOB LG_SOURCES src/*)

# Optimize these codes.
SET(CMAKE_CXX_FLAGS "-O2 ${CMAKE_CXX_FLAGS}")

function(add_app_folder folder)
    file(GLOB SOURCES "${folder}/*.cpp" ${LG_SOURCES})
    add_executable("${folder}" ${SOURCES})
    target_link_libraries("${folder}" Legion::Legion)
endfunction()

add_app_folder(summaMM)
