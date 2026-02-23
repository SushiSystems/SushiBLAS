#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../test_common.hpp"

class COPYTest : public SushiBLASTest {};

TEST_F(COPYTest, SimpleCOPY) 
{
    const int N = 3;
    auto X = engine->create_tensor({N});
    auto Y = engine->create_tensor({N});

    fill_tensor(X, {1, 2, 3});
    fill_tensor(Y, {0, 0, 0});

    SB_LOG_INFO("Submitting COPY.");
    engine->blas().copy(X, Y);
    
    engine->execute().wait();
    SB_LOG_INFO("Execution finished. Checking results.");

    // Expected: Y = X
    verify_tensor(Y, {1, 2, 3});
}
