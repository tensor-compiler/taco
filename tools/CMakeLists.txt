file(GLOB TOOL_SOURCES "*.cpp")

foreach(TOOL_SOURCE ${TOOL_SOURCES})
  get_filename_component(TOOL ${TOOL_SOURCE} NAME_WE)
  add_executable("${TOOL}-tool" ${TOOL_SOURCE})
  target_link_libraries("${TOOL}-tool" taco)
  if("${TOOL}" MATCHES "taco")
    target_include_directories(${TOOL}-tool
      PRIVATE
        ${PROJECT_SOURCE_DIR}/src
        ${CMAKE_BINARY_DIR}/include
    )
  endif()
  SET_TARGET_PROPERTIES("${TOOL}-tool" PROPERTIES OUTPUT_NAME ${TOOL})
  install(TARGETS "${TOOL}-tool" DESTINATION bin)
endforeach()

