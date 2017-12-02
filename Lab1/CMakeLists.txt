cmake_minimum_required(VERSION 2.8)
project( BespalovLab1 )
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
file(GLOB src
    "src/*.h"
    "src/*.cpp")
add_executable( BespalovLab1 ${src} )
