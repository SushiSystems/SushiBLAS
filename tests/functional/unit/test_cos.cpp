#include <gtest/gtest.h>
#include <vector>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"

using namespace SushiBLAS::Test;

class Unit_CosTest : public SushiBLASTest {};

TEST_F(Unit_CosTest, FloatForward) 
{
    auto t = engine->create_tensor({4});
    fill_tensor(t, {1.0f, 2.0f, -3.0f, 0.5f});
    
    engine->elementwise().cos(t);
    engine->execute().wait();
    
    verify_tensor(t, {0.54030f, -0.41615f, -0.98999f, 0.87758f});
}
