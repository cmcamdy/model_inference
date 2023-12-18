/* Copyright 2020 The TensorPilot Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
=============================================================================*/

#ifndef KERNEL_UTILS_ERROR_H_
#define KERNEL_UTILS_ERROR_H_

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#define LOGI(...)        \
  do {                   \
    printf("INFO:");     \
    printf(__VA_ARGS__); \
    printf("\n");        \
  } while (0)

#define LOGD(...)                                    \
  do {                                               \
    printf("DEBUG: %s:%d ", __FUNCTION__, __LINE__); \
    printf(__VA_ARGS__);                             \
    printf("\n");                                    \
  } while (0)

#define LOGE(...)                                    \
  do {                                               \
    printf("ERROR: %s:%d ", __FUNCTION__, __LINE__); \
    printf(__VA_ARGS__);                             \
    printf("\n");                                    \
  } while (0)

#define _FAIL(prefix, str)                                                     \
  do {                                                                         \
    std::cout << prefix << " : " << str << " " << __FILE__ << "  " << __LINE__ \
              << std::endl;                                                    \
    std::exit(-1);                                                             \
  } while (0)

#define CHECK_COND(cond, str)     \
  if (!(cond)) {                  \
    _FAIL("Check failed! ", str); \
  }

#endif  // !KERNEL_UTILS_ERROR_H_
