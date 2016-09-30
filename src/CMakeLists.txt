if (TAC_SHARED_LIBRARY)
  set(TAC_LIBRARY_TYPE SHARED)
  message("-- Shared library")
else()
  set(TAC_LIBRARY_TYPE STATIC)
  message("-- Static library")
endif()

set(TAC_SOURCE_DIRS . util)

foreach(dir ${TAC_SOURCE_DIRS})
  file(GLOB TAC_HEADERS ${TAC_HEADERS} ${dir}/*.h)
  file(GLOB TAC_SOURCES ${TAC_SOURCES} ${dir}/*.cpp)
endforeach()

set(TAC_HEADERS ${TAC_HEADERS})
set(TAC_SOURCES ${TAC_SOURCES})

add_definitions(${TAC_DEFINITIONS})
include_directories(${TAC_INCLUDE_DIRS})
add_library(taco ${TAC_LIBRARY_TYPE} ${TAC_HEADERS} ${TAC_SOURCES})
target_link_libraries(taco PRIVATE ${TAC_LIBRARIES})
