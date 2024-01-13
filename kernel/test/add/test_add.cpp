
#include <gtest/gtest.h>

#include <iostream>

#include "kernel/add/add.h"
#include "kernel/customop.h"
using namespace std;

// int main(){
//     cout << "1+2=" << add(1,2) << endl;
// }

TEST(AdditionTest, PositiveNumbers) {
  EXPECT_EQ(add(1, 2), 3);
  EXPECT_EQ(add(5, 7), 12);
}