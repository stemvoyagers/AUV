# Placeholder for the C++ navigation module.
# This tree is owned by the navigation workstream; the simulator does not build it.
cmake_minimum_required(VERSION 3.16)
project(auv_navigation CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Sources are added here by the navigation workstream.
file(GLOB NAVIGATION_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

if(NAVIGATION_SOURCES)
    add_library(auv_navigation STATIC ${NAVIGATION_SOURCES})
    target_include_directories(auv_navigation PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
endif()
