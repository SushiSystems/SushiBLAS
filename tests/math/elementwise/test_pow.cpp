#include <gtest/gtest.h>
#include <vector>
#include <SushiBLAS/SushiBLAS.h>
#include "../../test_common.hpp"

class PowTest : public SushiBLASTest {};

TEST_F(PowTest, FloatForward) 
{
    auto t = engine->create_tensor({4});
    fill_tensor(t, {1.0f, 2.0f, 3.0f, 0.5f});
    
    engine->elementwise().pow(t, 2.0f);
    engine->execute().wait();
    
    verify_tensor(t, {1.0f, 4.0f, 9.0f, 0.25f});
}
