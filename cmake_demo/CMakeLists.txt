cmake_minimum_required(VERSION 3.15)
project(cmake_demo)
add_compile_options(-std=c++17)

# set(CKLIB_DIR "" CACHE STRING "Location of local CKLIB_DIR.")
#　message(CKLIB_DIR: ${CKLIB_DIR})

# include_directories(${CKLIB_DIR}/include)
# find_package(hip REQUIRED PATHS /opt/rocm)
# find_package(hip REQUIRED PATHS /opt/rocm)
# message(STATUS "Build with HIP ${hip_VERSION}")

find_package(Threads REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/utils)
include_directories(${PROJECT_SOURCE_DIR}/include)


add_subdirectory(src)
add_subdirectory(example)



