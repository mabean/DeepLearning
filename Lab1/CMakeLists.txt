cmake_minimum_required(VERSION 2.8)
project( BespalovLab1 )
file(GLOB src
    "src/*.h"
    "src/*.cpp")
add_executable( BespalovLab1 ${src} )
