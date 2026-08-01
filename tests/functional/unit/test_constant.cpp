#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"

using namespace SushiBLAS::Test;

class Unit_ConstantTest : public SushiBLASTest {};

TEST_F(Unit_ConstantTest, Float32) 
{
    const int N = 10;
    auto t = engine->create_tensor({N});
    engine->random().constant(t, 2.5);
    engine->execute().wait();
    verify_tensor(t, std::vector<float>(N, 2.5f));
}
