#include <gtest/gtest.h>
#include <vector>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"

class Unit_AllTest : public SushiBLASTest {};

TEST_F(Unit_AllTest, FloatForward) 
{
    auto t = engine->create_tensor({4});
    fill_tensor(t, {1.0f, 0.0f, 3.0f, 0.0f});
    auto out = engine->create_tensor({1});
    engine->logic().all(t, out);
    engine->execute().wait();
    
    verify_tensor(out, {0.0f});
}
