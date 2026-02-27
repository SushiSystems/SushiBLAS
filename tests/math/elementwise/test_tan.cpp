#include <gtest/gtest.h>
#include <vector>
#include <SushiBLAS/SushiBLAS.h>
#include "../../test_common.hpp"

class TanTest : public SushiBLASTest {};

TEST_F(TanTest, FloatForward) 
{
    auto t = engine->create_tensor({4});
    fill_tensor(t, {1.0f, 2.0f, -3.0f, 0.5f});
    
    engine->elementwise().tan(t);
    engine->execute().wait();
    
    verify_tensor(t, {1.55741f, -2.18504f, 0.14255f, 0.54630f});
}
