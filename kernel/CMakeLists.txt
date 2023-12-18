cmake_minimum_required(VERSION 3.15)
project(Kernel)
add_compile_options(-std=c++17)

# includ cmake
include(cmake/3rdparty.cmake)


find_package(Threads REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/utils)
include_directories(${PROJECT_SOURCE_DIR}/include)

add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(example)



