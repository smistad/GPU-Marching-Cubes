cmake_minimum_required(VERSION 2.8)
project(GPU_MC)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

add_subdirectory(OpenCLUtilities)
include_directories(${OPENCL_INCLUDE_DIRS})

find_package(GLEW REQUIRED)
include_directories(${GLEW_INCLUDE_DIR})

find_package(OpenGL REQUIRED)
include_directories(${OPENGL_INCLUDE_DIR})

find_package(GLUT REQUIRED)
include_directories(${GLUT_INCLUDE_DIR})

add_executable(host main.cpp gpu-mc.cpp rawUtilities.cpp)

target_link_libraries(host OCL-Utilities ${GLEW_LIBRARY} ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES} )

