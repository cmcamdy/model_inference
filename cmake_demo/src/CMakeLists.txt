

add_library(kernel
    add/add.cc
)

target_include_directories(kernel
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)