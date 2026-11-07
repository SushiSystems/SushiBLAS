/**************************************************************************/
/* test_execution_mode.cpp                                                */
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
 * @file test_execution_mode.cpp
 * @brief Tests how the engine's execution mode selects and refuses kernels.
 * @author Mustafa Garip
 */

#include <vector>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_comparison.hpp>
#include <SushiBLAS/tensor_expectations/tensor_fill.hpp>
using namespace SushiBLAS;
using namespace SushiBLAS::Test;

/// @brief Fixture for the execution-mode tests.
class Medium_ExecutionModeTest : public SushiBLASTest
{

};

/**
 * @brief Records y = alpha * x + y under @p mode, executes it and returns the engine's y.
 * @param engine The engine to record on.
 */
static Tensor run_axpy(Engine& engine, ExecutionMode mode)
{
    engine.set_mode(mode);
    auto x = engine.create_tensor({4});
    auto y = engine.create_tensor({4});
    fill_tensor(x, {1.0f, 2.0f, 3.0f, 4.0f});
    fill_tensor(y, {10.0f, 20.0f, 30.0f, 40.0f});

    RecordingFixture fixture(TestEnvironment::runtime(), engine);
    SushiBLAS::BLASOps(fixture.recorder()).axpy(2.0, x, y);
    fixture.run().value();
    return y;
}

TEST_F(Medium_ExecutionModeTest, SetModeIsReadBack)
{
    engine->set_mode(ExecutionMode::ForceReference);

    EXPECT_EQ(ExecutionMode::ForceReference, engine->mode());
}

TEST_F(Medium_ExecutionModeTest, ForceReferenceAndForceSYCLAgreeOnAxpy)
{
    // 2 * {1, 2, 3, 4} + {10, 20, 30, 40}.
    const std::vector<float> expected = {12.0f, 24.0f, 36.0f, 48.0f};

    const Tensor reference = run_axpy(*engine, ExecutionMode::ForceReference);
    expect_tensor(reference, expected, absolute_tolerance<float>(2));

    const Tensor device = run_axpy(*engine, ExecutionMode::ForceSYCL);
    expect_tensor(device, expected, absolute_tolerance<float>(2));
}
