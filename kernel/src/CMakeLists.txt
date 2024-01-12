

add_library(kernel
    add/add.cc
    vector_add_1d/vector_add_1d.cc
)

target_include_directories(kernel
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)