/**************************************************************************/
/* host_engine_test.hpp                                                   */
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
 * @file host_engine_test.hpp
 * @brief Declares HostEngineTest, the fixture of a Small_ test of an operation: a host engine and no device.
 * @author Mustafa Garip
 * @see modules/engine/engine_fixture/README.md
 */

#pragma once

#include <memory>
#include <vector>
#include <cstring>
#include <cstdint>
#include <initializer_list>
#include <gtest/gtest.h>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/engine_fixture/host_recording_fixture.hpp>
#include <SushiBLAS/tensor_expectations/scalar_dtype.hpp>
#include <SushiBLAS/tensor_expectations/tensor_fill.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /**
         * @brief Gives each test a fresh host Engine and a HostRecordingFixture over it.
         *
         * It makes no queue and reaches no TestEnvironment: tensors are host memory, and
         * fill_tensor, fill_random, read_tensor and expect_tensor copy them by memcpy.
         * @see modules/engine/engine_fixture/README.md
         */
        class HostEngineTest : public ::testing::Test
        {
            protected:
                /// @brief The host engine of this test.
                std::unique_ptr<Engine> engine;

                /// @brief The recorder a test builds its families on, and its run.
                std::unique_ptr<HostRecordingFixture> recording;

                /** @brief Builds a row-major host engine and restarts the generator of fill_random. */
                void SetUp() override
                {
                    engine = std::make_unique<Engine>(Engine::Host{}, Core::Layout::RowMajor);
                    recording = std::make_unique<HostRecordingFixture>(*engine);
                    reset_fill_random_generator();
                }

                /** @brief Drops the recorder, then the engine. */
                void TearDown() override
                {
                    recording.reset();
                    engine.reset();
                }

                /** @brief Rebuilds the engine in another layout, and the recorder with it. */
                void reinit_engine(Core::Layout layout)
                {
                    recording.reset();
                    engine = std::make_unique<Engine>(Engine::Host{}, layout);
                    recording = std::make_unique<HostRecordingFixture>(*engine);
                }

                /** @brief Returns the recorder of recording, the one a test builds its families on. */
                TaskRecorder& recorder() noexcept { return recording->recorder(); }

                /** @brief Creates a tensor of dtype `T` and copies a host vector into it. */
                template<typename T>
                Tensor create_initialized_tensor(std::initializer_list<int64_t> dims,
                                                 const std::vector<T>& data)
                {
                    Tensor tensor = engine->create_tensor(dims, sushiblas_dtype<T>());
                    std::memcpy(tensor.data(), data.data(), data.size() * sizeof(T));
                    return tensor;
                }
        };
    } // namespace Test
} // namespace SushiBLAS
