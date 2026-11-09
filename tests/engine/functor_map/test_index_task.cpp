/**************************************************************************/
/* test_index_task.cpp                                                    */
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

/**
 * @file test_index_task.cpp
 * @brief Tests that an index task gives one result on the reference and the SYCL path, and that a late count is read when the host body runs.
 * @author Mustafa Garip
 */

#include <string>
#include <numeric>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/engine/dispatch.hpp>
#include <SushiBLAS/engine/execution_mode.hpp>
#include <SushiBLAS/engine/kernel_pair_dispatcher.hpp>
#include <SushiBLAS/functor_map/index_task.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief The capacity of every task of this file; not a multiple of any vector width.
        constexpr std::size_t K_CAPACITY = 1027;

        /** @brief Writes i + 1 into out[i] for the index it is called with. */
        struct NumberTheIndex
        {
            /** @brief The array written. */
            float* out;

            /** @brief Writes one element. */
            void operator()(std::size_t i) const
            {
                out[i] = static_cast<float>(i + 1);
            }
        };

        /** @brief Runs the tasks on the host CPU queue. */
        class Medium_IndexTask : public HostDeviceTest
        {
            protected:
                /** @brief Returns the array after a task over K_CAPACITY indices ran under mode. */
                std::vector<float> run_under(ExecutionMode mode)
                {
                    SharedArray<float> out(queue(), std::vector<float>(K_CAPACITY, 0.0f));

                    Dispatch::Detail::IndexTask<Dispatch::Domain::All, NumberTheIndex>(
                        Core::DataType::Float32, "test", "index", K_CAPACITY,
                        NumberTheIndex{out.get()})(KernelPairDispatcher(mode, queue()), {}).wait();

                    return out.values();
                }
        };

        TEST_F(Medium_IndexTask, BothPathsOfAFixedCountWriteEveryIndex)
        {
            const std::vector<float> by_reference = run_under(ExecutionMode::ForceReference);
            const std::vector<float> by_sycl = run_under(ExecutionMode::ForceSYCL);

            EXPECT_EQ(by_sycl, by_reference);
            EXPECT_EQ(by_reference.front(), 1.0f);
            EXPECT_EQ(by_reference.back(), static_cast<float>(K_CAPACITY));
        }

        /** @brief Returns K_CAPACITY zeros with 1, 2, ... written into the first count of them. */
        std::vector<float> numbered_prefix(std::size_t count)
        {
            std::vector<float> expected(K_CAPACITY, 0.0f);
            std::iota(expected.begin(), expected.begin() + static_cast<std::ptrdiff_t>(count), 1.0f);
            return expected;
        }

        /** @brief Runs the host body of a late-count index task over plain host memory. */
        class Small_LateIndexHostRun : public ::testing::Test
        {
            protected:
                /** @brief Returns the array after the host body ran over it with the width live. */
                static std::vector<float> run(std::uint32_t width)
                {
                    std::vector<float> out(K_CAPACITY, 0.0f);
                    const std::uint32_t live = width;

                    const Dispatch::Detail::LateIndexHostRun<NumberTheIndex> body(
                        NumberTheIndex{out.data()}, "index", K_CAPACITY, &live);
                    body();

                    return out;
                }
        };

        TEST_F(Small_LateIndexHostRun, TheHostBodyStopsAtTheWidth)
        {
            EXPECT_EQ(run(7), numbered_prefix(7));
        }

        TEST_F(Small_LateIndexHostRun, TheHostBodyReadsTheWidthWhenItRuns)
        {
            std::vector<float> out(K_CAPACITY, 0.0f);
            std::uint32_t live = 2;
            const Dispatch::Detail::LateIndexHostRun<NumberTheIndex> body(
                NumberTheIndex{out.data()}, "index", K_CAPACITY, &live);

            live = 5;
            body();

            EXPECT_EQ(out, numbered_prefix(5));
        }

        namespace
        {
            /** @brief Names a width the host body of a late count refuses. */
            struct RefusedWidthRow
            {
                const char* name;     /**< The row name a failure shows. */
                std::uint32_t width;  /**< The width the body reads. */
            };

            /** @brief Returns one row per width outside [0, capacity]. */
            std::vector<RefusedWidthRow> refused_width_rows()
            {
                // A negative INT32 count is the word 0xFFFFFFFF, far above any capacity.
                return {{"NegativeCount", static_cast<std::uint32_t>(std::int32_t{-1})},
                        {"OneAboveTheCapacity", static_cast<std::uint32_t>(K_CAPACITY + 1)}};
            }
        } // namespace

        /** @brief Runs one row of refused_width_rows(). */
        class Small_LateIndexHostRunRefusal
            : public Small_LateIndexHostRun,
              public ::testing::WithParamInterface<RefusedWidthRow>
        {
        };

        TEST_P(Small_LateIndexHostRunRefusal, TheHostBodyRefusesAWidthOutsideTheCapacity)
        {
            EXPECT_THROW(run(GetParam().width), std::runtime_error);
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_LateIndexHostRunRefusal, ::testing::ValuesIn(refused_width_rows()),
            [](const ::testing::TestParamInfo<RefusedWidthRow>& info)
            { return std::string(info.param.name); });
    } // namespace Test
} // namespace SushiBLAS
