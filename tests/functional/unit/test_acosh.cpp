#include <gtest/gtest.h>
#include <vector>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"

using namespace SushiBLAS::Test;

class Unit_AcoshTest : public SushiBLASTest {};

TEST_F(Unit_AcoshTest, FloatForward) 
{
    auto t = engine->create_tensor({4});
    fill_tensor(t, {1.0f, 2.0f, 3.0f, 4.0f});
    
    engine->elementwise().acosh(t);
    engine->execute().wait();
    
    verify_tensor(t, {0.00000f, 1.316957f, 1.762747f, 2.063437f});
}
