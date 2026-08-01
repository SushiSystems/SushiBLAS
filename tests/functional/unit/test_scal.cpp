#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"

class Unit_SCALTest : public SushiBLASTest {};

TEST_F(Unit_SCALTest, SimpleSCAL) 
{
    const int N = 3;
    auto X = engine->create_tensor({N});

    fill_tensor(X, {1, 2, 3});

    SB_LOG_INFO("Submitting SCAL.");
    engine->blas().scal(3.0f, X);
    
    engine->execute().wait();
    SB_LOG_INFO("Execution finished. Checking results.");

    // Expected: 3 * X = {3, 6, 9}
    verify_tensor(X, {3, 6, 9});
}
