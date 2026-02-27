#include <gtest/gtest.h>
#include <vector>
#include <SushiBLAS/SushiBLAS.h>
#include "../../test_common.hpp"

class DivTest : public SushiBLASTest {};

TEST_F(DivTest, FloatForward) 
{
    auto a = engine->create_tensor({4});
    auto b = engine->create_tensor({4});
    auto c = engine->create_tensor({4});
    fill_tensor(a, {1.0f, 2.0f, -3.0f, 0.5f});
    fill_tensor(b, {2.0f, 0.5f, 4.0f, -1.5f});
    
    engine->elementwise().div(a, b, c);
    engine->execute().wait();
    
    verify_tensor(c, {0.50000f, 4.00000f, -0.75000f, -0.33333f});
}
