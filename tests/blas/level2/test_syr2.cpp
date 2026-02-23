#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../test_common.hpp"

class SYR2Test : public SushiBLASTest {};

TEST_F(SYR2Test, SimpleSYR2) 
{
    const int N = 3;
    auto X = engine->create_tensor({N});
    auto Y = engine->create_tensor({N});
    auto A = engine->create_tensor({N, N});

    fill_tensor(X, {1, 2, 3});
    fill_tensor(Y, {1, 0, 0}); // Simplified for easy checking
    fill_tensor(A, std::vector<float>(N * N, 0.0f));

    SB_LOG_INFO("Submitting SYR2.");
    // upper=false
    engine->blas().syr2(X, Y, A, false);
    
    engine->execute().wait();
    SB_LOG_INFO("Execution finished. Checking results.");

    // Expected: Lower triangle of A = A + alpha * x * y^T + alpha * y * x^T
    // x * y^T:
    // [1 0 0]
    // [2 0 0]
    // [3 0 0]
    // y * x^T:
    // [1 2 3]
    // [0 0 0]
    // [0 0 0]
    //
    // Summing both gives:
    // [2 2 3]
    // [2 0 0]
    // [3 0 0]
    // Lower triangle update will only update the (i >= j) values from the theoretical sum.
    // Resulting matrix representation:
    // [2 0 0]
    // [2 0 0]
    // [3 0 0]
    
    verify_tensor(A, {2, 0, 0, 2, 0, 0, 3, 0, 0});
}
