#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../test_common.hpp"

class ROTTest : public SushiBLASTest {};

TEST_F(ROTTest, SimpleROT) 
{
    const int N = 2;
    auto X = engine->create_tensor({N});
    auto Y = engine->create_tensor({N});

    fill_tensor(X, {1, 0});
    fill_tensor(Y, {0, 1});

    // Rotate by 90 degrees: c = 0, s = 1
    float c = 0.0f;
    float s = 1.0f;

    SB_LOG_INFO("Submitting ROT.");
    engine->blas().rot(X, Y, c, s);
    
    engine->execute().wait();
    SB_LOG_INFO("Execution finished. Checking results.");

    // x_i = c*x_i + s*y_i => 0*1 + 1*0 = 0, 0*0 + 1*1 = 1
    // y_i = c*y_i - s*x_i => 0*0 - 1*1 = -1, 0*1 - 1*0 = 0
    verify_tensor(X, {0, 1});
    verify_tensor(Y, {-1, 0});
}
