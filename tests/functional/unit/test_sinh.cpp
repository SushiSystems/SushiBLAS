#include <gtest/gtest.h>
#include <vector>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"

using namespace SushiBLAS::Test;

class Unit_SinhTest : public SushiBLASTest {};

TEST_F(Unit_SinhTest, FloatForward) 
{
    auto t = engine->create_tensor({4});
    fill_tensor(t, {1.0f, 2.0f, -3.0f, 0.5f});
    
    engine->elementwise().sinh(t);
    engine->execute().wait();
    
    verify_tensor(t, {1.17520f, 3.62686f, -10.01787f, 0.52110f});
}
