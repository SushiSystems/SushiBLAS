#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../test_common.hpp"

class NormalTest : public SushiBLASTest {};

TEST_F(NormalTest, Float32) 
{
    const int N = 100;
    auto t = engine->create_tensor({N});
    engine->random().normal(t, 0.0, 1.0);
    engine->execute().wait();
    EXPECT_EQ(t.num_elements, N);
}
