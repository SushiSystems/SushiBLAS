#include <gtest/gtest.h>
#include <vector>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"

using namespace SushiBLAS::Test;

class Unit_AnyTest : public SushiBLASTest {};

TEST_F(Unit_AnyTest, FloatForward) 
{
    auto t = engine->create_tensor({4});
    fill_tensor(t, {1.0f, 0.0f, 3.0f, 0.0f});
    auto out = engine->create_tensor({1});
    engine->logic().any(t, out);
    engine->execute().wait();
    
    verify_tensor(out, {1.0f});
}
