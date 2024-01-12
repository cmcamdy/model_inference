#pragma once
#include <stdlib.h>

#include <string>

namespace kernel
{

    namespace test
    {

        std::string GetTestDataPath()
        {
            char *var = getenv("OPS_TEST_DATA_DIR");
            if (var)
            {
                return std::string(var);
            }
            else
            {
                return std::string();
            }
        }
    } // namespace test

} // namespace kernel
