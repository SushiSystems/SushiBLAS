/**************************************************************************/
/* runtime_port_calls.hpp                                                 */
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
 * @file runtime_port_calls.hpp
 * @brief Declares the calls the runtime tests make: each primitive recorded through a RuntimePrimitiveRecorder over a Recorder.
 * @author Mustafa Garip
 */

#pragma once

#include <utility>

#include <SushiRuntime/api/graph/recorder.hpp>
#include <SushiRuntime/core/result.hpp>

#include <SushiBLAS/graph_primitives/counting_sort.hpp>
#include <SushiBLAS/graph_primitives/prefix_scan.hpp>
#include <SushiBLAS/graph_primitives/reduce.hpp>
#include <SushiBLAS/graph_primitives/runtime_primitive_recorder.hpp>
#include <SushiBLAS/graph_primitives/segmented_reduce.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Holds the primitives with the runtime adapter built around the recorder given. */
        namespace RuntimePort
        {
            /** @brief Records counting_sort() through a RuntimePrimitiveRecorder over recorder. */
            template <typename... Arguments>
            [[nodiscard]] SushiRuntime::Result<void>
            counting_sort(SushiRuntime::Recorder& recorder, Arguments&&... arguments)
            {
                GraphPrimitives::RuntimePrimitiveRecorder port(recorder);
                return GraphPrimitives::counting_sort(port, std::forward<Arguments>(arguments)...);
            }

            /** @brief Records prefix_scan() through a RuntimePrimitiveRecorder over recorder. */
            template <typename... Arguments>
            [[nodiscard]] SushiRuntime::Result<void>
            prefix_scan(SushiRuntime::Recorder& recorder, Arguments&&... arguments)
            {
                GraphPrimitives::RuntimePrimitiveRecorder port(recorder);
                return GraphPrimitives::prefix_scan(port, std::forward<Arguments>(arguments)...);
            }

            /** @brief Records reduce() through a RuntimePrimitiveRecorder over recorder. */
            template <typename... Arguments>
            [[nodiscard]] SushiRuntime::Result<void>
            reduce(SushiRuntime::Recorder& recorder, Arguments&&... arguments)
            {
                GraphPrimitives::RuntimePrimitiveRecorder port(recorder);
                return GraphPrimitives::reduce(port, std::forward<Arguments>(arguments)...);
            }

            /** @brief Records segmented_reduce() through a RuntimePrimitiveRecorder over recorder. */
            template <typename... Arguments>
            [[nodiscard]] SushiRuntime::Result<void>
            segmented_reduce(SushiRuntime::Recorder& recorder, Arguments&&... arguments)
            {
                GraphPrimitives::RuntimePrimitiveRecorder port(recorder);
                return GraphPrimitives::segmented_reduce(port,
                                                         std::forward<Arguments>(arguments)...);
            }
        } // namespace RuntimePort
    } // namespace Test
} // namespace SushiBLAS
