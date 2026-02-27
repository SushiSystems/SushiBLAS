#include <gtest/gtest.h>
#include <vector>
#include <SushiBLAS/SushiBLAS.h>
#include "../../test_common.hpp"

class AsinTest : public SushiBLASTest {};

TEST_F(AsinTest, FloatForward) 
{
    auto t = engine->create_tensor({4});
    fill_tensor(t, {0.0f, 0.5f, -0.5f, 1.0f});
    
    engine->elementwise().asin(t);
    engine->execute().wait();
    
    verify_tensor(t, {0.00000f, 0.523598f, -0.523598f, 1.570796f});
}
