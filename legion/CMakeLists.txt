find_package(Legion REQUIRED)

# include_directories(legion legion/runtime)
add_executable(legion-test test-main.cpp)
target_link_libraries(legion-test Legion::Legion)