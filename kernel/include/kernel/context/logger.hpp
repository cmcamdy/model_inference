/*
 *Summary: print info(warning error info)
 */
#pragma once

#include <stdint.h>

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#define _LOG_INFO                                                              \
  do {                                                                         \
    std::cout << "INFO: "                                                      \
              << "Func:" << __FUNCTION__ << " line:" << __LINE__ << std::endl; \
  } while (0)

#define _LOG_WARNING                                                           \
  do {                                                                         \
    std::cout << "WARNING: "                                                   \
              << "Func:" << __FUNCTION__ << " line:" << __LINE__ << std::endl; \
  } while (0)

#define _LOG_ERROR                                                             \
  do {                                                                         \
    std::cout << "ERROR: "                                                     \
              << "Func:" << __FUNCTION__ << " line:" << __LINE__ << std::endl; \
  } while (0)

#define _LOG_FATAL                                                             \
  do {                                                                         \
    std::cout << "FATAL: "                                                     \
              << "Func:" << __FUNCTION__ << " line:" << __LINE__ << std::endl; \
    abort();                                                                   \
  } while (0)

#define LOG(severity) _LOG_##severity

#define CHECK(a) \
  if (!(a)) {    \
    LOG(ERROR);  \
  }
