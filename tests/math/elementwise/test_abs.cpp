#include <gtest/gtest.h>
#include <vector>
#include <SushiBLAS/SushiBLAS.h>
#include "../../test_common.hpp"

class AbsTest : public SushiBLASTest {};

TEST_F(AbsTest, FloatForward) 
{
    auto t = engine->create_tensor({4});
    fill_tensor(t, {1.0f, 2.0f, -3.0f, 0.5f});
    
    engine->elementwise().abs(t);
    engine->execute().wait();
    
    verify_tensor(t, {1.00000f, 2.00000f, 3.00000f, 0.50000f});
}
