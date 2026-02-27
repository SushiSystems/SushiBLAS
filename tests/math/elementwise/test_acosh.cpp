#include <gtest/gtest.h>
#include <vector>
#include <SushiBLAS/SushiBLAS.h>
#include "../../test_common.hpp"

class AcoshTest : public SushiBLASTest {};

TEST_F(AcoshTest, FloatForward) 
{
    auto t = engine->create_tensor({4});
    fill_tensor(t, {1.0f, 2.0f, 3.0f, 4.0f});
    
    engine->elementwise().acosh(t);
    engine->execute().wait();
    
    verify_tensor(t, {0.00000f, 1.316957f, 1.762747f, 2.063437f});
}
