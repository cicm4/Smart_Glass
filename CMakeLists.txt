cmake_minimum_required(VERSION 3.10)
project(Smart_Glass)

# Check for environment variable first
if(DEFINED ENV{OpenCV_DIR})
  message(STATUS "Using OpenCV_DIR from environment: $ENV{OpenCV_DIR}")
  set(OpenCV_DIR $ENV{OpenCV_DIR})
endif()

# Try to find OpenCV
find_package(OpenCV QUIET)

# If not found, provide better guidance
if(NOT OpenCV_FOUND)
  message(STATUS "OpenCV not found in standard locations")
  message(STATUS "Please set the OpenCV_DIR environment variable to your OpenCV build directory")
  message(STATUS "Example (Windows): set OpenCV_DIR=C:/path/to/opencv/build")
  message(STATUS "Example (Linux/macOS): export OpenCV_DIR=/path/to/opencv/build")
  message(STATUS "You can also pass it directly to CMake with: cmake -DOpenCV_DIR=/path/to/opencv/build ..")
  message(FATAL_ERROR "OpenCV not found. Aborting.")
endif()

# Include and link
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(Smart_Glass main.cpp)
target_link_libraries(Smart_Glass ${OpenCV_LIBS})