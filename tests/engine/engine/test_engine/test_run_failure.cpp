/**************************************************************************/
/* test_run_failure.cpp                                                   */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
/**************************************************************************/

/**
 * @file test_run_failure.cpp
 * @brief Tests what Engine::execute returns and leaves behind when a task body throws.
 * @author Mustafa Garip
 */

#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <SushiBLAS/SushiBLAS.hpp>

#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>

#include <SushiBLAS/tensor_expectations/tensor_comparison.hpp>

#include <SushiBLAS/tensor_expectations/tensor_fill.hpp>
using namespace SushiBLAS;
using namespace SushiBLAS::Test;

namespace
{
    /// @brief Holds the sentence the throwing body throws.
    constexpr const char* K_BODY_MESSAGE = "the body refused to run";

    /** @brief Records a task that declares a write of target and throws when it runs. */
    void record_throwing_task(Engine& engine, const Tensor& target)
    {
        using namespace SushiRuntime::Literals;

        TaskDescription description;
        description.name = "test.throwing_body";
        description.operation = "test.throwing_body"_op;

        OperandSet operands;
        operands.writes(target);

        engine.recorder().record_pair(
            description, operands, "Test operation",
            [](const auto&, sycl::queue&, const std::vector<sycl::event>&) -> sycl::event
            {
                throw std::runtime_error(K_BODY_MESSAGE);
            });
    }
} // namespace

/// @brief Fixture for the tests of a run that fails.
class Unit_RunFailureTest : public SushiBLASTest
{

};

/** @brief Tests that a body that throws makes execute return KernelFailed with its message. */
TEST_F(Unit_RunFailureTest, AThrowingBodyReturnsKernelFailedWithItsMessage)
{
    auto y = engine->create_tensor({4});
    record_throwing_task(*engine, y);

    const SushiRuntime::Result<void> ran = engine->execute();

    ASSERT_FALSE(ran.has_value());
    EXPECT_EQ(SushiRuntime::ErrorCode::KernelFailed, ran.error().code);
    EXPECT_NE(std::string::npos, ran.error().message.find(K_BODY_MESSAGE))
        << "message: " << ran.error().message;
    EXPECT_NE(std::string::npos, ran.error().message.find("test.throwing_body"))
        << "message: " << ran.error().message;
}

/** @brief Tests that a failed run leaves no task for the next execute to run. */
TEST_F(Unit_RunFailureTest, AFailedRunLeavesNothingRecorded)
{
    auto y = engine->create_tensor({4});
    fill_tensor(y, {10.0f, 20.0f, 30.0f, 40.0f});
    record_throwing_task(*engine, y);
    ASSERT_FALSE(engine->execute().has_value());

    // A throwing task still on the graph would fail this run too.
    EXPECT_TRUE(engine->execute().has_value());
    verify_tensor(y, {10.0f, 20.0f, 30.0f, 40.0f}, 0.0);
}

/** @brief Tests that the engine records and runs a valid operation after a failed run. */
TEST_F(Unit_RunFailureTest, TheEngineRunsAgainAfterAFailedRun)
{
    auto x = engine->create_tensor({4});
    auto y = engine->create_tensor({4});
    fill_tensor(x, {1.0f, 2.0f, 3.0f, 4.0f});
    fill_tensor(y, {10.0f, 20.0f, 30.0f, 40.0f});
    record_throwing_task(*engine, y);
    ASSERT_FALSE(engine->execute().has_value());

    BLASOps(*engine).axpy(2.0, x, y);
    const SushiRuntime::Result<void> ran = engine->execute();

    ASSERT_TRUE(ran.has_value()) << ran.error().message;
    // 2 * {1, 2, 3, 4} + {10, 20, 30, 40}.
    verify_tensor(y, {12.0f, 24.0f, 36.0f, 48.0f}, 0.0);
}
