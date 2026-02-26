#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../test_common.hpp"

class UniformTest : public SushiBLASTest {};

TEST_F(UniformTest, Float32) 
{
    const int N = 100;
    auto t = engine->create_tensor({N});
    engine->random().uniform(t, 0.0, 1.0);
    engine->execute().wait();
    const float* ptr = t.data_as<float>();
    for (int i = 0; i < N; ++i) {
        EXPECT_GE(ptr[i], 0.0f);
        EXPECT_LE(ptr[i], 1.0f);
    }
}
