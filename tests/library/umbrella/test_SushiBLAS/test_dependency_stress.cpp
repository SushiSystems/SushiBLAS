/**************************************************************************/
/* test_dependency_stress.cpp                                             */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file test_dependency_stress.cpp
 * @brief Tests the dependency tracker under shard collisions.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <vector>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_comparison.hpp>
#include <SushiBLAS/tensor_expectations/tensor_fill.hpp>
namespace sb = SushiBLAS;
using namespace SushiBLAS::Test;

namespace
{

constexpr int K_OPERATION_COUNT = 3000;
constexpr int K_SIDE = 8;
constexpr float K_B_VALUE = 0.5f;

/** @brief Gives the A value of operation @p index; it cycles through 251 distinct values. */
float a_value_of(int index)
{
    return static_cast<float>(index % 251 + 1);
}

/** @brief Holds the operands of independent GEMMs on distinct tensors. */
struct GemmBatch
{
    std::vector<sb::Tensor> a;
    std::vector<sb::Tensor> b;
    std::vector<sb::Tensor> c;
};

/** @brief Builds @p count filled A and B tensors and empty C tensors on @p engine. */
GemmBatch make_gemm_batch(sb::Engine& engine, int count)
{
    GemmBatch batch;
    for (int i = 0; i < count; ++i)
    {
        batch.a.push_back(engine.create_tensor({K_SIDE, K_SIDE}));
        batch.b.push_back(engine.create_tensor({K_SIDE, K_SIDE}));
        batch.c.push_back(engine.create_tensor({K_SIDE, K_SIDE}));
        fill_tensor(batch.a[i], std::vector<float>(K_SIDE * K_SIDE, a_value_of(i)));
        fill_tensor(batch.b[i], std::vector<float>(K_SIDE * K_SIDE, K_B_VALUE));
    }
    return batch;
}

/** @brief Expects every C of @p batch to hold K_SIDE * a * K_B_VALUE, exact in float. */
void expect_batch_results(GemmBatch const& batch)
{
    for (std::size_t i = 0; i < batch.c.size(); ++i)
    {
        expect_tensor_equal(batch.c[i], std::vector<float>(K_SIDE * K_SIDE,
                                                           K_SIDE * a_value_of(static_cast<int>(i)) * K_B_VALUE));
    }
}

} // namespace

class Medium_DependencyStressTest : public SushiBLASTest {};

/**
 * @brief Tests that 3000 independent GEMMs on distinct tensors each produce their own result.
 * @see tests/library/umbrella/README.md
 */
TEST_F(Medium_DependencyStressTest, ShardCollisionStress)
{
    GemmBatch batch = make_gemm_batch(*engine, K_OPERATION_COUNT);
    for (int i = 0; i < K_OPERATION_COUNT; ++i)
    {
        SushiBLAS::BLASOps(recorder()).gemm(batch.a[i], batch.b[i], batch.c[i]);
    }

    recording->run().value();

    expect_batch_results(batch);
}
