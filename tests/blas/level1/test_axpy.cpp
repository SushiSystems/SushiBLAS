#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../test_common.hpp"

class AXPYTest : public SushiBLASTest {};

TEST_F(AXPYTest, SimpleAXPY) 
{
    const int N = 3;
    auto X = engine->create_tensor({N});
    auto Y = engine->create_tensor({N});

    fill_tensor(X, {1, 2, 3});
    fill_tensor(Y, {4, 5, 6});

    SB_LOG_INFO("Submitting AXPY.");
    engine->blas().axpy(2.0f, X, Y);
    
    engine->execute().wait();
    SB_LOG_INFO("Execution finished. Checking results.");

    // Expected: 2 * X + Y = {2+4, 4+5, 6+6} = {6, 9, 12}
    verify_tensor(Y, {6, 9, 12});
}
