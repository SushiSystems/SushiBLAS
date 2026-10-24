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
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */

/**
 * @file test_index_task.cpp
 * @brief Tests that an index task gives one result on the reference and the SYCL path, and that a late count is read when the host body runs.
 * @author Mustafa Garip
 */

#include <string>
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
                        NumberTheIndex{out.get()})(KernelPairDispatcher(mode), queue(), {}).wait();

                    return out.values();
                }

                /** @brief Returns the array after the host body of a late count ran over it. */
                std::vector<float> run_late_host_body(std::uint32_t width)
                {
                    SharedArray<float> out(queue(), std::vector<float>(K_CAPACITY, 0.0f));
                    const std::uint32_t live = width;

                    const Dispatch::Detail::LateIndexHostRun<NumberTheIndex> body(
                        NumberTheIndex{out.get()}, "index", K_CAPACITY, &live);
                    body();

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

        TEST_F(Medium_IndexTask, TheHostBodyOfALateCountStopsAtTheWidth)
        {
            const std::vector<float> written = run_late_host_body(7);

            for (std::size_t i = 0; i < written.size(); ++i)
                EXPECT_EQ(written[i], i < 7 ? static_cast<float>(i + 1) : 0.0f) << "index " << i;
        }

        TEST_F(Medium_IndexTask, TheHostBodyOfALateCountReadsTheWidthWhenItRuns)
        {
            SharedArray<float> out(queue(), std::vector<float>(K_CAPACITY, 0.0f));
            std::uint32_t live = 2;
            const Dispatch::Detail::LateIndexHostRun<NumberTheIndex> body(
                NumberTheIndex{out.get()}, "index", K_CAPACITY, &live);

            live = 5;
            body();

            const std::vector<float> written = out.values();
            EXPECT_EQ(written[4], 5.0f);
            EXPECT_EQ(written[5], 0.0f);
        }

        TEST_F(Medium_IndexTask, TheHostBodyOfALateCountRefusesAWidthOutsideTheCapacity)
        {
            // A negative INT32 count is the word 0xFFFFFFFF, far above any capacity.
            EXPECT_THROW(run_late_host_body(static_cast<std::uint32_t>(std::int32_t{-1})),
                         std::runtime_error);
            EXPECT_THROW(run_late_host_body(static_cast<std::uint32_t>(K_CAPACITY + 1)),
                         std::runtime_error);
        }
    } // namespace Test
} // namespace SushiBLAS
