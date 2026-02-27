#include <gtest/gtest.h>
#include <vector>
#include <SushiBLAS/SushiBLAS.h>
#include "../../test_common.hpp"

class LogTest : public SushiBLASTest {};

TEST_F(LogTest, FloatForward) 
{
    auto t = engine->create_tensor({4});
    fill_tensor(t, {1.0f, 2.0f, 3.0f, 4.0f});
    
    engine->elementwise().log(t);
    engine->execute().wait();
    
    verify_tensor(t, {0.00000f, 0.693147f, 1.098612f, 1.386294f});
}
