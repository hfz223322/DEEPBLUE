cmake_minimum_required(VERSION 3.10.2)
project(active_contour)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-o2")
set(CMAKE_CXX_FLAGS "-Wall -Wextra")


find_package(OpenCV 3.2.0 REQUIRED)

find_package( Eigen3 REQUIRED)
include_directories("/usr/include/eigen3")

MESSAGE ( STATUS "OpenCV library status:" )
MESSAGE ( STATUS "version ${OpenCV_VERSION}" )
MESSAGE ( STATUS "libraries ${OpenCV_LIBS}" )
MESSAGE ( STATUS "include path ${OpenCV_INCLUDE_DIRS}" )

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)

add_library(grad_des_base ${CMAKE_SOURCE_DIR}/src/gradient_descent_base.cpp)

add_library(snake ${CMAKE_SOURCE_DIR}/src/snake.cpp)
target_link_libraries(snake ${OpenCV_LIBS} grad_des_base)

add_library(gvf ${CMAKE_SOURCE_DIR}/src/gvf.cpp)
target_link_libraries(gvf ${OpenCV_LIBS} grad_des_base)

add_library(display ${CMAKE_SOURCE_DIR}/src/display.cpp)
target_link_libraries(display snake ${OpenCV_LIBS})

add_executable(test_gvf ${CMAKE_SOURCE_DIR}/src/test_gvf.cpp)
target_link_libraries(test_gvf ${OpenCV_LIBS} gvf display)

add_executable(test_snake ${CMAKE_SOURCE_DIR}/src/test_snake.cpp)
target_link_libraries(test_snake ${OpenCV_LIBS} snake gvf display)

add_executable(use_graddescent ${CMAKE_SOURCE_DIR}/src/use_graddescent.cpp)
target_link_libraries(use_graddescent grad_des_base)
