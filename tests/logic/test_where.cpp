#include <gtest/gtest.h>
#include <vector>
#include <SushiBLAS/SushiBLAS.h>
#include "../test_common.hpp"

class WhereTest : public SushiBLASTest {};

TEST_F(WhereTest, FloatForward) 
{
    auto cond = engine->create_tensor({4});
    auto a = engine->create_tensor({4});
    auto b = engine->create_tensor({4});
    auto c = engine->create_tensor({4});
    fill_tensor(cond, {1.0f, 0.0f, 3.0f, 0.0f}); // act as cond
    fill_tensor(a, {1.0f, 0.0f, 3.0f, 0.0f});
    fill_tensor(b, {10.0f, 20.0f, 30.0f, 40.0f});
    engine->logic().where(cond, a, b, c);
    engine->execute().wait();
    
    verify_tensor(c, {1.00000f, 20.00000f, 3.00000f, 40.00000f});
}
