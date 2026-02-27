#include <gtest/gtest.h>
#include <vector>
#include <SushiBLAS/SushiBLAS.h>
#include "../../test_common.hpp"

class SqrtTest : public SushiBLASTest {};

TEST_F(SqrtTest, FloatForward) 
{
    auto t = engine->create_tensor({4});
    fill_tensor(t, {1.0f, 2.0f, 3.0f, 4.0f});
    
    engine->elementwise().sqrt(t);
    engine->execute().wait();
    
    verify_tensor(t, {1.00000f, 1.414213f, 1.732050f, 2.00000f});
}
