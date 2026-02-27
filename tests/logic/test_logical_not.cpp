#include <gtest/gtest.h>
#include <vector>
#include <SushiBLAS/SushiBLAS.h>
#include "../test_common.hpp"

class Logical_notTest : public SushiBLASTest {};

TEST_F(Logical_notTest, FloatForward) 
{
    auto a = engine->create_tensor({4});
    auto c = engine->create_tensor({4});
    fill_tensor(a, {1.0f, 0.0f, 3.0f, 0.0f});
    engine->logic().logical_not(a, c);
    engine->execute().wait();
    
    verify_tensor(c, {0.00000f, 1.00000f, 0.00000f, 1.00000f});
}
