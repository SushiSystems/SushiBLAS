#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../test_common.hpp"

class SYMVTest : public SushiBLASTest {};

TEST_F(SYMVTest, SimpleSYMV) 
{
    const int N = 3;
    auto A = engine->create_tensor({N, N});
    auto X = engine->create_tensor({N});
    auto Y = engine->create_tensor({N});

    // A (Symmetric, we only define upper/lower depending on flag. Let's make it full to test easily):
    // [1 2 3]
    // [2 4 5]
    // [3 5 6]
    fill_tensor(A, {1, 2, 3, 2, 4, 5, 3, 5, 6});
    
    // X:
    fill_tensor(X, {1, 1, 1});

    // Y:
    fill_tensor(Y, {0, 0, 0});

    SB_LOG_INFO("Submitting SYMV.");
    // upper=false -> lower triangle used by symv
    engine->blas().symv(A, X, Y, false);
    
    engine->execute().wait();
    SB_LOG_INFO("Execution finished. Checking results.");

    // Expected: Y = A * X
    // [1+2+3, 2+4+5, 3+5+6] = [6, 11, 14]
    verify_tensor(Y, {6, 11, 14});
}
