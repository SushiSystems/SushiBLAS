#include <gtest/gtest.h>
#include <vector>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"

class Unit_ReciprocalTest : public SushiBLASTest {};

TEST_F(Unit_ReciprocalTest, FloatForward) 
{
    auto t = engine->create_tensor({4});
    fill_tensor(t, {1.0f, 2.0f, -3.0f, 0.5f});
    
    engine->elementwise().reciprocal(t);
    engine->execute().wait();
    
    verify_tensor(t, {1.00000f, 0.50000f, -0.33333f, 2.00000f});
}
