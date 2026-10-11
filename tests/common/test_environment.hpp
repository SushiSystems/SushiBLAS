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
 * @file test_environment.hpp
 * @brief Declares the GoogleTest environment that owns the shared runtime.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <optional>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiRuntime/sushiruntime.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /**
         * @brief Owns the one runtime every test of a binary shares.
         * @see tests/common/README.md
         */
        class TestEnvironment : public ::testing::Environment
        {
            public:
                /** @brief Creates the runtime; a runtime that cannot be created stops the process. */
                void SetUp() override
                {
                    SB_LOG_INFO("--- Initializing SushiBLAS Test Environment ---");
                    runtime_.emplace(SushiRuntime::Runtime::create().value());

                    SB_LOG_INFO("Test environment ready.");
                }

                /** @brief Destroys the runtime. */
                void TearDown() override
                {
                    SB_LOG_INFO("--- Tearing down SushiBLAS Test Environment ---");
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
                 * @see tests/common/README.md
                 */
                [[nodiscard]] static std::size_t live_allocation_count()
                {
                    return runtime_->unsafe_context().memory().live_count();
                }

            private:
                /// @brief The shared runtime; empty outside SetUp() and TearDown().
                static inline std::optional<SushiRuntime::Runtime> runtime_;
        };

    } // namespace Test
} // namespace SushiBLAS
