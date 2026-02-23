#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../test_common.hpp"

class DOTTest : public SushiBLASTest {};

TEST_F(DOTTest, SimpleDOT) 
{
    const int N = 3;
    auto X = engine->create_tensor({N});
    auto Y = engine->create_tensor({N});
    auto R = engine->create_tensor({1}); // Scalar result tensor

    fill_tensor(X, {1, 2, 3});
    fill_tensor(Y, {4, 5, 6});
    fill_tensor(R, {0.0f});

    SB_LOG_INFO("Submitting DOT.");
    engine->blas().dot(X, Y, R);
    
    engine->execute().wait();
    SB_LOG_INFO("Execution finished. Checking results.");

    // Expected: 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    verify_tensor(R, {32.0f});
}
