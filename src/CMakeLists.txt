if (TACO_SHARED_LIBRARY)
  set(TACO_LIBRARY_TYPE SHARED)
  message("-- Shared library")
else()
  set(TACO_LIBRARY_TYPE STATIC)
  message("-- Static library")
endif()

set(TACO_SOURCE_DIRS . storage lower backends util)

foreach(dir ${TACO_SOURCE_DIRS})
  file(GLOB TACO_HEADERS ${TACO_HEADERS} ${dir}/*.h)
  file(GLOB TACO_SOURCES ${TACO_SOURCES} ${dir}/*.cpp)
endforeach()

set(TACO_HEADERS ${TACO_HEADERS})
set(TACO_SOURCES ${TACO_SOURCES})

add_definitions(${TACO_DEFINITIONS})
include_directories(${TACO_INCLUDE_DIRS})
add_library(taco ${TACO_LIBRARY_TYPE} ${TACO_HEADERS} ${TACO_SOURCES})

if (LINUX)
  target_link_libraries(taco PRIVATE ${TACO_LIBRARIES} dl)
else()
  target_link_libraries(taco PRIVATE ${TACO_LIBRARIES})
endif()
