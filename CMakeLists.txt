cmake_minimum_required(VERSION 3.15)
project(videoCut)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_LIBS})

add_executable(videoCut main.cpp tools.h)
target_link_libraries(videoCut ${OpenCV_LIBS})