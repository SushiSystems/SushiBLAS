#include <gtest/gtest.h>
#include <vector>
#include <SushiBLAS/SushiBLAS.h>
#include "../../test_common.hpp"

class CoshTest : public SushiBLASTest {};

TEST_F(CoshTest, FloatForward) 
{
    auto t = engine->create_tensor({4});
    fill_tensor(t, {1.0f, 2.0f, -3.0f, 0.5f});
    
    engine->elementwise().cosh(t);
    engine->execute().wait();
    
    verify_tensor(t, {1.54308f, 3.76220f, 10.06766f, 1.12763f});
}
