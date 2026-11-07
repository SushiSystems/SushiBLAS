/**************************************************************************/
/* test_environment.hpp                                                   */
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
 * @file test_environment.hpp
 * @brief Declares the GoogleTest environment that owns the shared runtime and the host CPU queue.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <optional>
#include <gtest/gtest.h>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/test_runner/cpu_queue.hpp>
#include <SushiBLAS/test_runner/test_runner_log_category.hpp>
#include <SushiRuntime/sushiruntime.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /**
         * @brief Owns the one runtime and the one host CPU queue every test of a binary shares.
         * @see modules/foundation/test_runner/README.md
         */
        class TestEnvironment : public ::testing::Environment
        {
            public:
                /** @brief Creates the runtime and the host CPU queue, or stops the process if the runtime cannot be created. */
                void SetUp() override
                {
                    SB_LOG_INFO(K_TEST_RUNNER_LOG_CATEGORY, "--- Initializing SushiBLAS Test Environment ---");
                    runtime_.emplace(SushiRuntime::Runtime::create().value());
                    cpu_queue_ = Detail::make_cpu_queue();

                    SB_LOG_INFO(K_TEST_RUNNER_LOG_CATEGORY, "Test environment ready.");
                }

                /** @brief Destroys the host CPU queue and the runtime. */
                void TearDown() override
                {
                    SB_LOG_INFO(K_TEST_RUNNER_LOG_CATEGORY, "--- Tearing down SushiBLAS Test Environment ---");
                    cpu_queue_.reset();
                    runtime_.reset();
                }

                /**
                 * @brief Returns the shared runtime.
                 * @pre SetUp() has run and TearDown() has not.
                 */
                [[nodiscard]] static SushiRuntime::Runtime& runtime()
                {
                    return *runtime_;
                }

                /**
                 * @brief Returns the queue of one device of the shared runtime.
                 * @param device_index Below device_count().
                 */
                [[nodiscard]] static sycl::queue& queue(std::size_t device_index = 0)
                {
                    return runtime_->advanced().queue(SushiRuntime::DeviceIndex{device_index});
                }

                /** @brief Returns how many devices the shared runtime drives. */
                [[nodiscard]] static std::size_t device_count()
                {
                    return runtime_->advanced().device_count();
                }

                /**
                 * @brief Returns how many USM allocations of the shared runtime are live.
                 * @see modules/foundation/test_runner/README.md
                 */
                [[nodiscard]] static std::size_t live_allocation_count()
                {
                    return runtime_->advanced().live_allocation_count();
                }

                /**
                 * @brief Returns the in-order queue on the first CPU device SYCL reports.
                 * @return nullptr when SYCL reports no CPU device, or outside SetUp() and TearDown().
                 * @see modules/foundation/test_runner/README.md
                 */
                [[nodiscard]] static sycl::queue* cpu_queue()
                {
                    return cpu_queue_ ? &*cpu_queue_ : nullptr;
                }

            private:
                /// @brief The shared runtime; empty outside SetUp() and TearDown().
                static inline std::optional<SushiRuntime::Runtime> runtime_;

                /// @brief The host CPU queue; empty without a CPU device and outside SetUp() and TearDown().
                static inline std::optional<sycl::queue> cpu_queue_;
        };

    } // namespace Test
} // namespace SushiBLAS
