/**************************************************************************/
/* runtime_primitive_recorder.hpp                                         */
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
 * @file runtime_primitive_recorder.hpp
 * @brief Declares RuntimePrimitiveRecorder, the PrimitiveRecorder adapter over a SushiRuntime Recorder.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <cstddef>
#include <utility>

#include <SushiRuntime/api/data/buffer.hpp>
#include <SushiRuntime/api/graph/access.hpp>
#include <SushiRuntime/api/graph/device_sized.hpp>
#include <SushiRuntime/api/graph/node_options.hpp>
#include <SushiRuntime/api/graph/recorder.hpp>
#include <SushiRuntime/api/vocabulary/late_count.hpp>
#include <SushiRuntime/core/error.hpp>
#include <SushiRuntime/core/result.hpp>

#include <SushiBLAS/graph_primitives/primitive_recorder.hpp>

namespace SushiBLAS
{
    namespace GraphPrimitives
    {
        /**
         * @brief Forwards each call of the PrimitiveRecorder port to one SushiRuntime::Recorder.
         * @see modules/foundation/graph_primitives/README.md
         */
        class RuntimePrimitiveRecorder
        {
            public:
                /** @brief Marks this type as a PrimitiveRecorder. */
                static constexpr bool K_PRIMITIVE_RECORDER = true;

                /** @brief Guard of one rule; nodes recorded while it lives share the rule. */
                using RuleScope = SushiRuntime::Recorder::RuleScope;

                /** @brief Takes the recorder the calls are forwarded to; it must outlive this. */
                explicit RuntimePrimitiveRecorder(SushiRuntime::Recorder& recorder) noexcept
                    : recorder_(recorder)
                {
                }

                /** @brief Opens a rule; see SushiRuntime::Recorder::rule(). */
                [[nodiscard]] RuleScope rule() noexcept { return recorder_.rule(); }

                /** @brief Allocates a block of count elements the recorder owns. */
                template <typename T>
                [[nodiscard]] SushiRuntime::Result<T*>
                scratch(std::size_t count, SushiRuntime::DeviceIndex device,
                        SushiRuntime::Residency residency)
                {
                    return recorder_.scratch<T>(count, device, residency);
                }

                /** @brief Records the task that resolves counts for a run. */
                template <std::size_t K, class Fill>
                std::size_t resolve(const std::array<SushiRuntime::LateCount, K>& counts,
                                    SushiRuntime::DeviceIndex device,
                                    SushiRuntime::Access access, Fill fill,
                                    SushiRuntime::NodeOptions options = {})
                {
                    return recorder_.resolve(counts, device, std::move(access), std::move(fill),
                                             options);
                }

                /** @brief Records a launch of count work-items. */
                template <class F>
                std::size_t kernel(std::size_t count, SushiRuntime::Access access, F&& body,
                                   SushiRuntime::NodeOptions options = {})
                {
                    return recorder_.kernel(count, std::move(access), std::forward<F>(body),
                                            options);
                }

                /** @brief Records a launch of a device-sized capacity. */
                template <class F>
                std::size_t kernel(SushiRuntime::DeviceSized launch, SushiRuntime::Access access,
                                   F&& body, SushiRuntime::NodeOptions options = {})
                {
                    return recorder_.kernel(launch, std::move(access), std::forward<F>(body),
                                            options);
                }

                /** @brief Returns the first recording the recorder refused, or null. */
                [[nodiscard]] const SushiRuntime::Error* refusal() const
                {
                    return recorder_.refusal();
                }

            private:
                /** @brief The recorder every call is forwarded to. */
                SushiRuntime::Recorder& recorder_;
        };

    } // namespace GraphPrimitives
} // namespace SushiBLAS
