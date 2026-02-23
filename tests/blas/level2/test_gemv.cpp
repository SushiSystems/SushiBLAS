#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../test_common.hpp"

class GEMVTest : public SushiBLASTest {};

TEST_F(GEMVTest, SimpleGEMV) 
{
    const int M = 2, N = 3;
    auto A = engine->create_tensor({M, N});
    auto X = engine->create_tensor({N});
    auto Y = engine->create_tensor({M});

    // A:
    // [1 2 3]
    // [4 5 6]
    fill_tensor(A, {1, 2, 3, 4, 5, 6});
    
    // X:
    // [1, 1, 1]
    fill_tensor(X, {1, 1, 1});

    // Y:
    // [0, 0]
    fill_tensor(Y, {0, 0});

    SB_LOG_INFO("Submitting GEMV.");
    engine->blas().gemv(A, X, Y);
    
    engine->execute().wait();
    SB_LOG_INFO("Execution finished. Checking results.");

    // Expected: Y = A * X = [1+2+3, 4+5+6] = [6, 15]
    verify_tensor(Y, {6, 15});
}
