#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../test_common.hpp"

class TRSVTest : public SushiBLASTest {};

TEST_F(TRSVTest, SimpleTRSV) 
{
    const int N = 3;
    auto A = engine->create_tensor({N, N});
    auto b = engine->create_tensor({N});

    // A (Lower triangular):
    // [1 0 0]
    // [2 1 0]
    // [3 4 1]
    fill_tensor(A, {1, 0, 0, 2, 1, 0, 3, 4, 1});
    
    // b: (results from multiplying A by x = [1, 1, 1] which is [1, 3, 8])
    fill_tensor(b, {1, 3, 8});

    SB_LOG_INFO("Submitting TRSV.");
    // upper=false, transA=false, unit_diag=false
    engine->blas().trsv(A, b, false, false, false);
    
    engine->execute().wait();
    SB_LOG_INFO("Execution finished. Checking results.");

    // Expected: b solved to x = [1, 1, 1]
    verify_tensor(b, {1, 1, 1});
}
