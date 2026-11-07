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
 * @brief Tests the partial sizes, the lease rule and an index reduction recorded on an engine.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/operation_id.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/engine/execution_mode.hpp>
#include <SushiBLAS/reduction/reduce_1d.hpp>
#include <SushiBLAS/reductions/arg_reduction.hpp>
#include <SushiBLAS/reductions/dispatch.hpp>
#include <SushiBLAS/engine_fixture/host_engine_test.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/graph_probe/graph_probe.hpp>
#include <SushiBLAS/tensor_expectations/tensor_readback.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief The bytes of an IEEE binary16 value, the width a half partial must exceed.
        constexpr std::size_t K_HALF_BYTES = 2;

        /// @brief The bytes of a FLOAT64 partial.
        constexpr std::size_t K_FLOAT64_BYTES = 8;

        /// @brief The bytes of an INT32 partial.
        constexpr std::size_t K_INT32_BYTES = 4;

        /** @brief Records an argmax of t into result through recorder. */
        void record_argmax(TaskRecorder& recorder, const Tensor& t, Tensor& result)
        {
            Dispatch::execute_index_reduction(
                recorder, t, result, "argmax", SushiRuntime::OperationID(),
                Dispatch::ArgReductionDispatch<Reduce::RunningArgMax>{},
                Dispatch::planned_partials_for(t));
        }

        /** @brief Records reductions on a host engine, which runs ForceReference only. */
        class Small_ReductionDispatchOnHost : public HostEngineTest
        {
        };

        /** @brief Records reductions on the fixture's engine under a mode of its own. */
        class Medium_ReductionDispatch : public SushiBLASTest
        {
            protected:
                /** @brief Returns the index argmax records and runs under mode on a fixed input. */
                std::vector<int64_t> argmax_index_under(ExecutionMode mode)
                {
                    const Tensor t = create_initialized_tensor<float>({5}, {2, 7, -3, 7, -3});
                    Tensor result = create_initialized_tensor<int64_t>({1}, {-1});

                    set_mode(mode);
                    record_argmax(recorder(), t, result);
                    EXPECT_TRUE(recording->run().has_value());

                    return read_tensor<int64_t>(result);
                }
        };

        TEST(Small_ReductionDispatch, AHalfPartialIsWiderThanTheHalfItSums)
        {
            EXPECT_GT(Dispatch::partial_bytes(Core::DataType::Half), K_HALF_BYTES);
        }

        TEST(Small_ReductionDispatch, AFloat64PartialTakesEightBytes)
        {
            EXPECT_EQ(Dispatch::partial_bytes(Core::DataType::Float64), K_FLOAT64_BYTES);
        }

        TEST(Small_ReductionDispatch, AnInt32PartialTakesFourBytes)
        {
            EXPECT_EQ(Dispatch::partial_bytes(Core::DataType::Int32), K_INT32_BYTES);
        }

        TEST_F(Small_ReductionDispatchOnHost, AnEmptyTensorPlansNoPartials)
        {
            EXPECT_EQ(Dispatch::planned_partials_for(engine->create_tensor({0})), std::size_t{0});
        }

        TEST_F(Small_ReductionDispatchOnHost, TheReferenceRecorderLeasesNoPartials)
        {
            EXPECT_EQ(Dispatch::leased_partials(recorder(), 12), std::size_t{0});
        }

        TEST_F(Medium_ReductionDispatch, TheAutomaticRecorderLeasesThePlannedPartials)
        {
            GraphProbe automatic;

            EXPECT_EQ(Dispatch::leased_partials(automatic.recorder(), 12), std::size_t{12});
        }

        TEST_F(Small_ReductionDispatchOnHost, AnIndexReductionFindsTheFirstLargestOnTheReference)
        {
            const Tensor t = create_initialized_tensor<float>({5}, {2, 7, -3, 7, -3});
            Tensor result = create_initialized_tensor<int64_t>({1}, {-1});

            record_argmax(recorder(), t, result);
            ASSERT_TRUE(recording->run_succeeds());

            EXPECT_EQ(read_tensor<int64_t>(result), (std::vector<int64_t>{1}));
        }

        TEST_F(Medium_ReductionDispatch, AnIndexReductionFindsTheFirstLargestOnSYCL)
        {
            EXPECT_EQ(argmax_index_under(ExecutionMode::ForceSYCL), (std::vector<int64_t>{1}));
        }
    } // namespace Test
} // namespace SushiBLAS
