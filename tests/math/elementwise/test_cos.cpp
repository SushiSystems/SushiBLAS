#include <gtest/gtest.h>
#include <vector>
#include <SushiBLAS/SushiBLAS.h>
#include "../../test_common.hpp"

class CosTest : public SushiBLASTest {};

TEST_F(CosTest, FloatForward) 
{
    auto t = engine->create_tensor({4});
    fill_tensor(t, {1.0f, 2.0f, -3.0f, 0.5f});
    
    engine->elementwise().cos(t);
    engine->execute().wait();
    
    verify_tensor(t, {0.54030f, -0.41615f, -0.98999f, 0.87758f});
}
