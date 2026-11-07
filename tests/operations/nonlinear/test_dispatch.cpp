/**************************************************************************/
/* test_dispatch.cpp                                                      */
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
 * @file test_dispatch.cpp
 * @brief Tests a forward task in place and a backward task with a bound scalar in both modes.
 * @author Mustafa Garip
 */

#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/operation_id.hpp>
#include <SushiBLAS/engine/execution_mode.hpp>
#include <SushiBLAS/functor_map/scalar.hpp>
#include <SushiBLAS/functor_map/scalar_binding.hpp>
#include <SushiBLAS/nonlinear/dispatch.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/execution_mode_sweep/execution_mode_sweep.hpp>
#include <SushiBLAS/tensor_expectations/tensor_readback.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Scales a negative element by a and keeps the rest. */
        struct ScaleNegative
        {
            /** @brief Returns the result for one element. */
            template<typename T>
            T operator()(T x, T a) const
            {
                return x < T(0) ? x * a : x;
            }
        };

        /** @brief Passes the gradient where the input is positive and zero elsewhere. */
        struct PositiveGate
        {
            /** @brief Returns the gradient for one element. */
            template<typename T>
            T operator()(T g, T v) const
            {
                return v > T(0) ? g : T(0);
            }
        };

        /** @brief Records nonlinear tasks on the fixture's engine, once in each execution mode. */
        class Medium_NonlinearDispatch : public SushiBLASTest,
                                         public ::testing::WithParamInterface<ExecutionMode>
        {

        };

        TEST_P(Medium_NonlinearDispatch, ForwardRewritesTheTensorInPlace)
        {
            Tensor t = create_initialized_tensor<float>({4}, {-2, -1, 0, 3});
            set_mode(GetParam());

            Dispatch::execute_nonlinear_forward(
                recorder(), t, "scale_negative", SushiRuntime::OperationID(),
                Dispatch::bind_scalars(ScaleNegative{}, Dispatch::ElementScalar(0.5)), {0.5});
            ASSERT_TRUE(recording->run().has_value());

            EXPECT_EQ(read_tensor<float>(t), (std::vector<float>{-1, -0.5f, 0, 3}));
        }

        TEST_P(Medium_NonlinearDispatch, BackwardGatesTheGradientByTheInput)
        {
            const Tensor dy = create_initialized_tensor<double>({3}, {4, 5, 6});
            const Tensor x = create_initialized_tensor<double>({3}, {-1, 2, 0});
            Tensor dx = create_initialized_tensor<double>({3}, {9, 9, 9});
            set_mode(GetParam());

            Dispatch::execute_nonlinear_backward(recorder(), dy, x, dx, "gate",
                                                 SushiRuntime::OperationID(), PositiveGate{});
            ASSERT_TRUE(recording->run().has_value());

            EXPECT_EQ(read_tensor<double>(dx), (std::vector<double>{0, 5, 0}));
        }

        INSTANTIATE_TEST_SUITE_P(
            , Medium_NonlinearDispatch, ::testing::ValuesIn(K_BOTH_MODES),
            [](const ::testing::TestParamInfo<ExecutionMode>& info)
            { return std::string(info.param == ExecutionMode::ForceReference ? "ForceReference"
                                                                           : "ForceSYCL"); });
    } // namespace Test
} // namespace SushiBLAS
