#include <gtest/gtest.h>
#include <vector>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"

using namespace SushiBLAS::Test;

class Unit_SquareTest : public SushiBLASTest {};

TEST_F(Unit_SquareTest, FloatForward) 
{
    auto t = engine->create_tensor({4});
    fill_tensor(t, {1.0f, 2.0f, -3.0f, 0.5f});
    
    engine->elementwise().square(t);
    engine->execute().wait();
    
    verify_tensor(t, {1.00000f, 4.00000f, 9.00000f, 0.25000f});
}
