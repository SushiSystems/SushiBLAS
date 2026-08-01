#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"

class Unit_ASUMTest : public SushiBLASTest {};

TEST_F(Unit_ASUMTest, SimpleASUM) 
{
    const int N = 4;
    auto X = engine->create_tensor({N});
    auto R = engine->create_tensor({1}); // Scalar result tensor

    fill_tensor(X, {-1, 2, -3, 4});
    fill_tensor(R, {0.0f});

    SB_LOG_INFO("Submitting ASUM.");
    engine->blas().asum(X, R);
    
    engine->execute().wait();
    SB_LOG_INFO("Execution finished. Checking results.");

    // Expected asum: |-1| + |2| + |-3| + |4| = 1 + 2 + 3 + 4 = 10
    verify_tensor(R, {10.0f});
}
