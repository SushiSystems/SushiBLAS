#include <gtest/gtest.h>
#include <vector>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"

class Unit_AsinhTest : public SushiBLASTest {};

TEST_F(Unit_AsinhTest, FloatForward) 
{
    auto t = engine->create_tensor({4});
    fill_tensor(t, {1.0f, 2.0f, -3.0f, 0.5f});
    
    engine->elementwise().asinh(t);
    engine->execute().wait();
    
    verify_tensor(t, {0.88137f, 1.44364f, -1.81845f, 0.48121f});
}
