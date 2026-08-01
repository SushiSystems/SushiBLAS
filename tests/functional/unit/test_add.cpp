#include <gtest/gtest.h>
#include <vector>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"

class Unit_AddTest : public SushiBLASTest {};

TEST_F(Unit_AddTest, FloatForward) 
{
    auto a = engine->create_tensor({4});
    auto b = engine->create_tensor({4});
    auto c = engine->create_tensor({4});
    fill_tensor(a, {1.0f, 2.0f, -3.0f, 0.5f});
    fill_tensor(b, {2.0f, 0.5f, 4.0f, -1.5f});
    
    engine->elementwise().add(a, b, c);
    engine->execute().wait();
    
    verify_tensor(c, {3.00000f, 2.50000f, 1.00000f, -1.00000f});
}
