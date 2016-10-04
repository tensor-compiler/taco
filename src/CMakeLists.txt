if (TACIT_SHARED_LIBRARY)
  set(TACIT_LIBRARY_TYPE SHARED)
  message("-- Shared library")
else()
  set(TACIT_LIBRARY_TYPE STATIC)
  message("-- Static library")
endif()

set(TACIT_SOURCE_DIRS . util)

foreach(dir ${TACIT_SOURCE_DIRS})
  file(GLOB TACIT_HEADERS ${TACIT_HEADERS} ${dir}/*.h)
  file(GLOB TACIT_SOURCES ${TACIT_SOURCES} ${dir}/*.cpp)
endforeach()

set(TACIT_HEADERS ${TACIT_HEADERS})
set(TACIT_SOURCES ${TACIT_SOURCES})

add_definitions(${TACIT_DEFINITIONS})
include_directories(${TACIT_INCLUDE_DIRS})
add_library(tacit ${TACIT_LIBRARY_TYPE} ${TACIT_HEADERS} ${TACIT_SOURCES})
target_link_libraries(tacit PRIVATE ${TACIT_LIBRARIES})
