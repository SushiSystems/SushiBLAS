/**************************************************************************/
/* primitive_recorder_subjects.hpp                                        */
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
 * @file primitive_recorder_subjects.hpp
 * @brief Declares the subjects a PrimitiveRecorder contract runs on: a HostPrimitiveRecorder and a
 *        RuntimePrimitiveRecorder over a runtime graph.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include <SushiRuntime/api/data/buffer.hpp>
#include <SushiRuntime/api/graph/graph.hpp>
#include <SushiRuntime/api/runtime/runtime.hpp>
#include <SushiRuntime/core/result.hpp>

#include <SushiBLAS/graph_primitives/host_primitive_recorder.hpp>
#include <SushiBLAS/graph_primitives/runtime_primitive_recorder.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Holds the words of the output buffer every contract check writes.
        inline constexpr std::size_t K_SUBJECT_OUTPUT_WORDS = 8;

        /** @brief Records on a host recorder and runs the host forms in record order. */
        class HostPrimitiveSubject
        {
            public:
                /** @brief Builds the output buffer on the recorder; no runtime is made. */
                HostPrimitiveSubject()
                    : output_(GraphPrimitives::HostPrimitiveRecorder::buffer<std::uint32_t>(
                          K_SUBJECT_OUTPUT_WORDS))
                {
                }

                HostPrimitiveSubject(const HostPrimitiveSubject&) = delete;
                HostPrimitiveSubject& operator=(const HostPrimitiveSubject&) = delete;

                /** @brief Returns the recorder under test. */
                [[nodiscard]] GraphPrimitives::HostPrimitiveRecorder& port() noexcept
                {
                    return port_;
                }

                /** @brief Returns the words the checks write into. */
                [[nodiscard]] std::uint32_t* output() const noexcept { return output_.data(); }

                /** @brief Returns the device scratch is requested on. */
                [[nodiscard]] SushiRuntime::DeviceIndex device() const noexcept
                {
                    return output_.device_index();
                }

                /** @brief Returns the residency scratch is requested in. */
                [[nodiscard]] SushiRuntime::Residency residency() const noexcept
                {
                    return output_.residency();
                }

                /** @brief Returns how many rules the recorded nodes belong to. */
                [[nodiscard]] std::size_t rule_count() const noexcept
                {
                    return port_.rule_count();
                }

                /** @brief Runs the recorded nodes; the run must succeed. */
                void run() { port_.run().value(); }

            private:
                /// @brief The words the checks write into.
                GraphPrimitives::HostBuffer<std::uint32_t> output_;

                /// @brief The recorder under test.
                GraphPrimitives::HostPrimitiveRecorder port_;
        };

        /** @brief Records on a runtime graph through the runtime adapter and runs the graph. */
        class RuntimePrimitiveSubject
        {
            public:
                /** @brief Builds the output buffer and the graph on a runtime of its own. */
                RuntimePrimitiveSubject()
                    : runtime_(SushiRuntime::Runtime::create().value()),
                      output_(runtime_.buffer<std::uint32_t>(K_SUBJECT_OUTPUT_WORDS).value()),
                      graph_(runtime_.graph()),
                      port_(graph_.recorder())
                {
                }

                RuntimePrimitiveSubject(const RuntimePrimitiveSubject&) = delete;
                RuntimePrimitiveSubject& operator=(const RuntimePrimitiveSubject&) = delete;

                /** @brief Returns the recorder under test. */
                [[nodiscard]] GraphPrimitives::RuntimePrimitiveRecorder& port() noexcept
                {
                    return port_;
                }

                /** @brief Returns the words the checks write into. */
                [[nodiscard]] std::uint32_t* output() const noexcept { return output_.data(); }

                /** @brief Returns the device scratch is requested on. */
                [[nodiscard]] SushiRuntime::DeviceIndex device() const noexcept
                {
                    return output_.device_index();
                }

                /** @brief Returns the residency scratch is requested in. */
                [[nodiscard]] SushiRuntime::Residency residency() const noexcept
                {
                    return output_.residency();
                }

                /** @brief Returns how many rules the recorded nodes belong to. */
                [[nodiscard]] std::size_t rule_count() noexcept
                {
                    return graph_.recorder().rule_count();
                }

                /** @brief Runs the graph; the run must succeed. */
                void run() { graph_.run().value(); }

            private:
                /// @brief Owns the output buffer and the graph.
                SushiRuntime::Runtime runtime_;

                /// @brief The words the checks write into.
                SushiRuntime::Buffer<std::uint32_t> output_;

                /// @brief The graph the recorder belongs to.
                SushiRuntime::Graph graph_;

                /// @brief The recorder under test.
                GraphPrimitives::RuntimePrimitiveRecorder port_;
        };
    } // namespace Test
} // namespace SushiBLAS
