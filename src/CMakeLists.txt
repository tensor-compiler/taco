if (TACO_SHARED_LIBRARY)
  set(TACO_LIBRARY_TYPE SHARED)
  message("-- Shared library")
else()
  set(TACO_LIBRARY_TYPE STATIC)
  message("-- Static library")
endif()

set(TACO_SRC_DIRS . error expr_nodes lower ir storage backends util parser io)

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
  target_link_libraries(taco PRIVATE ${TACO_LIBRARIES} dl)
else()
  target_link_libraries(taco PRIVATE ${TACO_LIBRARIES})
endif()
