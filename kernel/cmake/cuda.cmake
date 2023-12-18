#--------------------------------------------------------------
# cuda configure
#--------------------------------------------------------------

message(STATUS "CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")
set(CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")

find_package(CUDA REQUIRED)

if(${CUDA_FOUND})
    set(CUDAToolkit_FOUND ON)
    set(CUDAToolkit_VERSION ${CUDA_VERSION})
    set(CUDAToolkit_NVCC_EXECUTABLE ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc)
    set(CMAKE_CUDA_COMPILER ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc)
    set(CUDAToolkit_INCLUDE_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/targets/${ARCH}-linux/include)
    set(CUDAToolkit_LIBRARY_DIR ${CUDA_TOOLKIT_ROOT_DIR}/targets/${ARCH}-linux/lib)

    link_directories(${CUDAToolkit_LIBRARY_DIR})
    link_directories(${CUDAToolkit_LIBRARY_DIR}/stubs)

    set(CUDA_DRIVER_LIB cuda)
    set(CUDA_RT_LIB cudart)
    set(CUBLAS_LIB cublas cublasLt)
endif()  # ${CUDA_FOUND}

if(${CUDAToolkit_FOUND})
    enable_language(CUDA)
    include_directories("${CUDAToolkit_INCLUDE_DIRS}")

    # set cuda flags
    if((DEFINED CMAKE_CUDA_FLAGS) AND (${CMAKE_CUDA_FLAGS} MATCHES "^.*-arch=sm_.*"))
        message(STATUS "cuda arch already set")
    else()
        if(${TARGET} MATCHES "^orin$") # orin
            message(STATUS "cuda arch not set, default set to sm_87 for orin.")
            set(CMAKE_CUDA_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_87")
        elseif(${TARGET} MATCHES "^x86_64$") # a100
            message(STATUS "cuda arch not set, default set to sm_80 for x86_64.")
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_80 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86")
        else()
            message(STATUS "cuda arch and platform arch not set, default set to sm_75.")
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_75")
        endif()
    endif()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fPIC,-fno-gnu-unique --expt-relaxed-constexpr --Werror default-stream-launch -ccbin ${CUDA_HOST_COMPILER}")
    if(${CUDA_VERSION} STRGREATER_EQUAL "11.4") # orin
        set(CMAKE_CUDA_FLAGS "-Xcudafe \"--display_error_number --diag_suppress=177 --diag_suppress=191 --diag_suppress=512 --diag_suppress=20012 --diag_suppress=20014\" ${CMAKE_CUDA_FLAGS}")
    elseif(${CUDA_VERSION} STRGREATER_EQUAL "11.0") # a100
        set(CMAKE_CUDA_FLAGS "-Xcudafe \"--display_error_number --diag_suppress=177 --diag_suppress=191 --diag_suppress=512 --diag_suppress=3059 --diag_suppress=esa_on_defaulted_function_ignored\" ${CMAKE_CUDA_FLAGS}")
    endif()

else()
    message(FATAL_ERROR "CUDA not found.")
endif()

#--------------------------------------------------------------
# print cuda variables
#--------------------------------------------------------------
# cuda variables
message(STATUS "CUDA_FOUND: ${CUDA_FOUND}")
message(STATUS "CUDA_VERSION: ${CUDA_VERSION}")
message(STATUS "CUDAToolkit_FOUND: ${CUDAToolkit_FOUND}")
message(STATUS "CUDAToolkit_VERSION: ${CUDAToolkit_VERSION}")
message(STATUS "CUDAToolkit_NVCC_EXECUTABLE: ${CUDAToolkit_NVCC_EXECUTABLE}")
message(STATUS "CMAKE_CUDA_COMPILER: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CUDAToolkit_INCLUDE_DIRS: ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "CUDAToolkit_LIBRARY_DIR: ${CUDAToolkit_LIBRARY_DIR}")

message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
message(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
message(STATUS "CUDA_NVCC_FLAGS: ${CUDA_NVCC_FLAGS}")
