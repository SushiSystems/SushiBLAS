#include <gtest/gtest.h>
#include <vector>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"

using namespace SushiBLAS::Test;

class Unit_ExpTest : public SushiBLASTest {};

TEST_F(Unit_ExpTest, FloatForward) 
{
    auto t = engine->create_tensor({4});
    fill_tensor(t, {1.0f, 2.0f, -3.0f, 0.5f});
    
    engine->elementwise().exp(t);
    engine->execute().wait();
    
    verify_tensor(t, {2.71828f, 7.38906f, 0.04979f, 1.64872f});
}
