#include <gtest/gtest.h>
#include <vector>
#include <SushiBLAS/SushiBLAS.h>
#include "../../test_common.hpp"

class RoundTest : public SushiBLASTest {};

TEST_F(RoundTest, FloatForward) 
{
    auto t = engine->create_tensor({4});
    fill_tensor(t, {1.1f, 2.9f, -3.2f, 0.0f});
    
    engine->elementwise().round(t);
    engine->execute().wait();
    
    verify_tensor(t, {1.00000f, 3.00000f, -3.00000f, 0.00000f});
}
