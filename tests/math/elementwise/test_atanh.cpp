#include <gtest/gtest.h>
#include <vector>
#include <SushiBLAS/SushiBLAS.h>
#include "../../test_common.hpp"

class AtanhTest : public SushiBLASTest {};

TEST_F(AtanhTest, FloatForward) 
{
    auto t = engine->create_tensor({4});
    fill_tensor(t, {0.0f, 0.5f, -0.5f, 0.8f});
    
    engine->elementwise().atanh(t);
    engine->execute().wait();
    
    verify_tensor(t, {0.00000f, 0.549306f, -0.549306f, 1.098612f});
}
