add_executable(test_add test_add.cpp)
target_link_libraries(test_add
    PUBLIC	
        kernel
        Threads::Threads
        ${GTEST_LIBS}
)
