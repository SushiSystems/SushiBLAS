#include <gtest/gtest.h>
#include <vector>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"

using namespace SushiBLAS::Test;

class Unit_FloorTest : public SushiBLASTest {};

TEST_F(Unit_FloorTest, FloatForward) 
{
    auto t = engine->create_tensor({4});
    fill_tensor(t, {1.0f, 2.0f, -3.0f, 0.5f});
    
    engine->elementwise().floor(t);
    engine->execute().wait();
    
    verify_tensor(t, {1.00000f, 2.00000f, -3.00000f, 0.00000f});
}
