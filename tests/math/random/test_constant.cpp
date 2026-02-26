#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../test_common.hpp"

class ConstantTest : public SushiBLASTest {};

TEST_F(ConstantTest, Float32) 
{
    const int N = 10;
    auto t = engine->create_tensor({N});
    engine->random().constant(t, 2.5);
    engine->execute().wait();
    verify_tensor(t, std::vector<float>(N, 2.5f));
}
