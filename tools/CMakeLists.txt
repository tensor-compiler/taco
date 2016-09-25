file(GLOB TOOL_SOURCES "${TAC_TOOLS_DIR}/*.cpp")

foreach(TOOL_SOURCE ${TOOL_SOURCES})
	get_filename_component(TOOL ${TOOL_SOURCE} NAME_WE)
	add_executable(${TOOL} ${TOOL_SOURCE})
	target_link_libraries(${TOOL} tac)
endforeach()
