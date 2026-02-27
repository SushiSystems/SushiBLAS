#include <gtest/gtest.h>
#include <vector>
#include <SushiBLAS/SushiBLAS.h>
#include "../../test_common.hpp"

class AtanTest : public SushiBLASTest {};

TEST_F(AtanTest, FloatForward) 
{
    auto t = engine->create_tensor({4});
    fill_tensor(t, {1.0f, 2.0f, -3.0f, 0.5f});
    
    engine->elementwise().atan(t);
    engine->execute().wait();
    
    verify_tensor(t, {0.78540f, 1.10715f, -1.24905f, 0.46365f});
}
