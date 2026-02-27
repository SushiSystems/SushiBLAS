#include <gtest/gtest.h>
#include <vector>
#include <SushiBLAS/SushiBLAS.h>
#include "../../test_common.hpp"

class AcosTest : public SushiBLASTest {};

TEST_F(AcosTest, FloatForward) 
{
    auto t = engine->create_tensor({4});
    fill_tensor(t, {0.0f, 0.5f, -0.5f, 1.0f});
    
    engine->elementwise().acos(t);
    engine->execute().wait();
    
    verify_tensor(t, {1.570796f, 1.047197f, 2.094395f, 0.00000f});
}
