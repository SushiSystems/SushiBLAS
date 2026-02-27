#include <gtest/gtest.h>
#include <vector>
#include <SushiBLAS/SushiBLAS.h>
#include "../../test_common.hpp"

class ClampTest : public SushiBLASTest {};

TEST_F(ClampTest, FloatForward) 
{
    auto t = engine->create_tensor({4});
    fill_tensor(t, {1.0f, 2.0f, -3.0f, 0.5f});
    
    engine->elementwise().clamp(t, -1.5f, 1.5f);
    engine->execute().wait();
    
    verify_tensor(t, {1.00000f, 1.50000f, -1.50000f, 0.50000f});
}
