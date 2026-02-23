#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../test_common.hpp"

class TRMVTest : public SushiBLASTest {};

TEST_F(TRMVTest, SimpleTRMV) 
{
    const int N = 3;
    auto A = engine->create_tensor({N, N});
    auto X = engine->create_tensor({N});

    // A (Lower triangular by default in this test because upper=false):
    // [1 0 0]   -> in row major representation [1, 2, 3] doesn't matter if we just fill array 
    // [2 4 0]      but let's set it as full and MKL will just read lower.
    // [3 5 6]
    fill_tensor(A, {1, 0, 0, 2, 4, 0, 3, 5, 6});
    
    // X:
    fill_tensor(X, {1, 1, 1});

    SB_LOG_INFO("Submitting TRMV.");
    // upper=false, transA=false, unit_diag=false
    engine->blas().trmv(A, X, false, false, false);
    
    engine->execute().wait();
    SB_LOG_INFO("Execution finished. Checking results.");

    // Expected: X = A * X
    // [1, 2+4, 3+5+6] = [1, 6, 14]
    verify_tensor(X, {1, 6, 14});
}
