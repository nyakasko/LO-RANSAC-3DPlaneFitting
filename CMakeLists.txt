# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.8)

project ("LSQ")
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( lsq LSQ/LSQ.cpp )
target_link_libraries( lsq ${OpenCV_LIBS} )
# Include sub-projects.
add_subdirectory ("LSQ")


