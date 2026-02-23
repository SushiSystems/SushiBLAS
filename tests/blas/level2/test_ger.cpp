#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../test_common.hpp"

class GERTest : public SushiBLASTest {};

TEST_F(GERTest, SimpleGER) 
{
    const int M = 2, N = 3;
    auto X = engine->create_tensor({M});
    auto Y = engine->create_tensor({N});
    auto A = engine->create_tensor({M, N});

    fill_tensor(X, {1, 2});
    fill_tensor(Y, {1, 2, 3});
    
    // Initial A is zeros
    fill_tensor(A, {0, 0, 0, 0, 0, 0});

    SB_LOG_INFO("Submitting GER.");
    engine->blas().ger(X, Y, A);
    
    engine->execute().wait();
    SB_LOG_INFO("Execution finished. Checking results.");

    // Expected: A = A + X * Y^T
    // [1] * [1, 2, 3] = [1, 2, 3]
    // [2]               [2, 4, 6]
    verify_tensor(A, {1, 2, 3, 2, 4, 6});
}
