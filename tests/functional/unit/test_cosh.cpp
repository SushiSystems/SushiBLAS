#include <gtest/gtest.h>
#include <vector>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"

class Unit_CoshTest : public SushiBLASTest {};

TEST_F(Unit_CoshTest, FloatForward) 
{
    auto t = engine->create_tensor({4});
    fill_tensor(t, {1.0f, 2.0f, -3.0f, 0.5f});
    
    engine->elementwise().cosh(t);
    engine->execute().wait();
    
    verify_tensor(t, {1.54308f, 3.76220f, 10.06766f, 1.12763f});
}
