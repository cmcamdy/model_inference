add_library(kernel
    add/add.cc
    vector_add_1d/vector_add_1d.cc
    reduce_sum/reduce_sum.cc
    reduce_2d_op/reduce_2d_op.cc
)

add_library(kernel_launch
    vector_add_1d/kernels/forwar_functor.cu
    reduce_sum/kernels/forwar_functor.cu
)

target_link_libraries(kernel
    kernel_launch
)

target_include_directories(kernel 
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)