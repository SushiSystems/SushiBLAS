#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../test_common.hpp"

class SYRTest : public SushiBLASTest {};

TEST_F(SYRTest, SimpleSYR) 
{
    const int N = 3;
    auto X = engine->create_tensor({N});
    auto A = engine->create_tensor({N, N});

    fill_tensor(X, {1, 2, 3});
    fill_tensor(A, std::vector<float>(N * N, 0.0f));

    SB_LOG_INFO("Submitting SYR.");
    // upper=false
    engine->blas().syr(X, A, false);
    
    engine->execute().wait();
    SB_LOG_INFO("Execution finished. Checking results.");

    // Expected: Lower triangle of A is updated by alpha * x * x^T 
    // x * x^T:
    // [1]           [1 2 3]
    // [2] * [1 2 3]= [2 4 6]
    // [3]           [3 6 9]
    //
    // Since only lower triangle is updated: (row-major)
    // [1 0 0]
    // [2 4 0]
    // [3 6 9]
    
    verify_tensor(A, {1, 0, 0, 2, 4, 0, 3, 6, 9});
}
