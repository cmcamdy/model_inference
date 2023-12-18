
# gtest
if(${IS_CROSS_COMPILER} OR ${TARGET} MATCHES "^orin$")
    set(GTEST_INCLUDE_DIRS ${DEPS_INCLUDE_DIR})
    set(GTEST_LIBS ${DEPS_DIR}/lib/libgtest.a ${DEPS_DIR}/lib/libgtest_main.a)
    set(GTEST_FOUND True)
    message(STATUS "GTEST FOUND: ${GTEST_FOUND}")
else()
    find_package(GTest)
    set(GTEST_LIBS ${GTEST_BOTH_LIBRARIES})
    message(STATUS "GTEST FOUND: ${GTEST_FOUND}")
    # message(STATUS "GTEST VERSION: ${GTEST_VERSION}")
    # message(STATUS "GTEST include: ${GTEST_INCLUDE_DIRS}")
    # message(STATUS "GTEST libs: ${GTEST_LIBS}")
    # message(STATUS "GTEST libraries: ${GTEST_LIBRARIES}")
    # message(STATUS "GTEST both libraries: ${GTEST_BOTH_LIBRARIES}")
endif()
