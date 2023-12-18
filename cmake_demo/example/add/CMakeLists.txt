add_executable(add test_add.cpp)
target_link_libraries(add
    PUBLIC	
        kernel
        Threads::Threads
)
