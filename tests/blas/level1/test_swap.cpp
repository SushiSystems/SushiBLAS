#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../test_common.hpp"

class SWAPTest : public SushiBLASTest {};

TEST_F(SWAPTest, SimpleSWAP) 
{
    const int N = 3;
    auto X = engine->create_tensor({N});
    auto Y = engine->create_tensor({N});

    fill_tensor(X, {1, 2, 3});
    fill_tensor(Y, {4, 5, 6});

    SB_LOG_INFO("Submitting SWAP.");
    engine->blas().swap(X, Y);
    
    engine->execute().wait();
    SB_LOG_INFO("Execution finished. Checking results.");

    // Expected: X = {4, 5, 6}, Y = {1, 2, 3}
    verify_tensor(X, {4, 5, 6});
    verify_tensor(Y, {1, 2, 3});
}
