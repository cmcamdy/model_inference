

file(GLOB_RECURSE TEST_SRCS_CU
../test_context/test_context.cu
)

file(GLOB_RECURSE TEST_SRCS_CC
../data_loader/cnpy.cpp
test_reduce_2d_op.cpp
)

add_library(test_reduce_2d_op_test_cuda OBJECT ${TEST_SRCS_CU})
add_executable(test_reduce_2d_op ${TEST_SRCS_CC} $<TARGET_OBJECTS:test_reduce_2d_op_test_cuda>)
target_link_libraries(test_reduce_2d_op
    PUBLIC	
        kernel
        Threads::Threads
        ${GTEST_LIBS}
        ${LIBZ_LIBS}
)
