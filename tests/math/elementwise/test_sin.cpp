#include <gtest/gtest.h>
#include <vector>
#include <SushiBLAS/SushiBLAS.h>
#include "../../test_common.hpp"

class SinTest : public SushiBLASTest {};

TEST_F(SinTest, FloatForward) 
{
    auto t = engine->create_tensor({4});
    fill_tensor(t, {1.0f, 2.0f, -3.0f, 0.5f});
    
    engine->elementwise().sin(t);
    engine->execute().wait();
    
    verify_tensor(t, {0.84147f, 0.90930f, -0.14112f, 0.47943f});
}
