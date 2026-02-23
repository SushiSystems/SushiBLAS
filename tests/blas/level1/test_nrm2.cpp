#include <vector>
#include <cmath>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../test_common.hpp"

class NRM2Test : public SushiBLASTest {};

TEST_F(NRM2Test, SimpleNRM2) 
{
    const int N = 3;
    auto X = engine->create_tensor({N});
    auto R = engine->create_tensor({1}); // Scalar result tensor

    fill_tensor(X, {3, 4, 0});
    fill_tensor(R, {0.0f});

    SB_LOG_INFO("Submitting NRM2.");
    engine->blas().nrm2(X, R);
    
    engine->execute().wait();
    SB_LOG_INFO("Execution finished. Checking results.");

    // Expected norm: sqrt(3^2 + 4^2 + 0^2) = sqrt(25) = 5
    verify_tensor(R, {5.0f});
}
