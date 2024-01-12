

file(GLOB_RECURSE TEST_SRCS_CU
../test_context/test_context.cu
)

file(GLOB_RECURSE TEST_SRCS_CC
../data_loader/cnpy.cpp
test_vector_add_1d.cpp
)

add_library(test_vector_add_1d_test_cuda OBJECT ${TEST_SRCS_CU})
add_executable(test_vector_add_1d ${TEST_SRCS_CC} $<TARGET_OBJECTS:test_vector_add_1d_test_cuda>)
target_link_libraries(test_vector_add_1d
    PUBLIC	
        kernel
        Threads::Threads
        ${GTEST_LIBS}
        ${LIBZ_LIBS}
)
