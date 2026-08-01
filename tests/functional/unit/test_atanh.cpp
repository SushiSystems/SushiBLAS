#include <gtest/gtest.h>
#include <vector>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"

class Unit_AtanhTest : public SushiBLASTest {};

TEST_F(Unit_AtanhTest, FloatForward) 
{
    auto t = engine->create_tensor({4});
    fill_tensor(t, {0.0f, 0.5f, -0.5f, 0.8f});
    
    engine->elementwise().atanh(t);
    engine->execute().wait();
    
    verify_tensor(t, {0.00000f, 0.549306f, -0.549306f, 1.098612f});
}
