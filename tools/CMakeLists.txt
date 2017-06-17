file(GLOB TOOL_SOURCES "${TACO_TOOLS_DIR}/*.cpp")

foreach(TOOL_SOURCE ${TOOL_SOURCES})
  get_filename_component(TOOL ${TOOL_SOURCE} NAME_WE)
  add_executable("${TOOL}-tool" ${TOOL_SOURCE})
  target_link_libraries("${TOOL}-tool" taco)
  SET_TARGET_PROPERTIES("${TOOL}-tool" PROPERTIES OUTPUT_NAME ${TOOL})
  install(TARGETS "${TOOL}-tool" DESTINATION bin)
endforeach()

include_directories(${TACO_SRC_DIR})
