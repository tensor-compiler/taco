set(TACO_SRC_DIRS . parser index_notation lower ir codegen storage error util)

foreach(dir ${TACO_SRC_DIRS})
  file(GLOB TACO_HEADERS ${TACO_HEADERS} ${dir}/*.h)
  file(GLOB TACO_HEADERS ${TACO_HEADERS} ${TACO_INCLUDE_DIR}/taco/${dir}/*.h)
  file(GLOB TACO_SOURCES ${TACO_SOURCES} ${dir}/*.cpp)
endforeach()

set(TACO_HEADERS ${TACO_HEADERS} ../include/taco/ir_tags.h)
set(TACO_SOURCES ${TACO_SOURCES} ir_tags.cpp)

add_library(taco ${TACO_LIBRARY_TYPE} ${TACO_HEADERS} ${TACO_SOURCES})
if (CUDA)
  include_directories(${CUDA_INCLUDE_DIRS})
  target_link_libraries(taco PUBLIC ${CUDA_LIBRARIES})
endif (CUDA)
target_include_directories(taco
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/taco
    ${CMAKE_BINARY_DIR}
  PRIVATE
    ${PROJECT_SOURCE_DIR}/src


)
target_link_libraries(taco
  PRIVATE
    $<TARGET_NAME_IF_EXISTS:OpenMP::OpenMP_CXX>
)
target_compile_definitions(taco
  PRIVATE
    "$<IF:$<TARGET_EXISTS:OpenMP::OpenMP_CXX>,USE_OPENMP,>"
    $<$<PLATFORM_ID:Linux>:dl>
)

install(TARGETS taco DESTINATION lib)

