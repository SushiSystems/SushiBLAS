#include <gtest/gtest.h>
#include <vector>
#include <SushiBLAS/SushiBLAS.h>
#include "../test_common.hpp"

class Less_equalTest : public SushiBLASTest {};

TEST_F(Less_equalTest, FloatForward) 
{
    auto a = engine->create_tensor({4});
    auto b = engine->create_tensor({4});
    auto c = engine->create_tensor({4});
    fill_tensor(a, {1.0f, 0.0f, 3.0f, 0.0f});
    fill_tensor(b, {1.0f, 1.0f, 0.0f, 0.0f});
    
    engine->logic().less_equal(a, b, c);
    engine->execute().wait();
    
    verify_tensor(c, {1.00000f, 1.00000f, 0.00000f, 1.00000f});
}
