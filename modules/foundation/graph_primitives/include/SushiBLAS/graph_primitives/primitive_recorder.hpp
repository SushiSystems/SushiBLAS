/**************************************************************************/
/* primitive_recorder.hpp                                                 */
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
 * @file primitive_recorder.hpp
 * @brief Declares the PrimitiveRecorder port and the record a host adapter keeps of each node.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <SushiRuntime/core/determinism.hpp>
#include <SushiRuntime/core/operation_id.hpp>
#include <SushiRuntime/region/buffer_region.hpp>

namespace SushiBLAS
{
    namespace GraphPrimitives
    {
        /**
         * @brief Names the kind of node a primitive records.
         * @see modules/foundation/graph_primitives/README.md
         */
        enum class PrimitiveNodeKind
        {
            Kernel,  ///< A launch of capacity work-items.
            Resolve  ///< The one task that resolves the late counts of a run.
        };

        /**
         * @brief Holds what a host adapter keeps of one node a primitive recorded.
         * @see modules/foundation/graph_primitives/README.md
         */
        struct PrimitiveNodeRecord
        {
            PrimitiveNodeKind kind = PrimitiveNodeKind::Kernel; /**< Kind of the node. */
            std::size_t rule = 0;                    /**< Rule the node belongs to, from 0. */
            std::string name;                        /**< Name the call stated; empty if none. */
            SushiRuntime::Determinism determinism =
                SushiRuntime::Determinism::Unstated; /**< Class the call stated. */
            SushiRuntime::OperationID operation;     /**< Hash of the label of the node. */
            std::size_t width = 0;                   /**< Launch width stamped into the node. */
            std::size_t capacity = 0;                /**< Work-items launched, or late counts resolved. */
            std::vector<std::size_t> late_capacities; /**< Capacity of each late count of a resolve. */
            bool has_live_width = false;             /**< Whether the launch reads a live item count. */
            bool has_live_words = false;             /**< Whether the launch reads live words. */
            std::vector<SushiRuntime::BufferRegion> reads;  /**< Regions the node reads. */
            std::vector<SushiRuntime::BufferRegion> writes; /**< Regions the node writes. */
        };

        /**
         * @brief Tells whether Port declares itself an adapter of the PrimitiveRecorder port.
         *
         * An adapter sets a static constexpr K_PRIMITIVE_RECORDER to true. The port is a
         * compile-time contract because a kernel body stays a device-copyable type.
         * @see modules/foundation/graph_primitives/README.md
         */
        template <class Port, class = void>
        inline constexpr bool K_IS_PRIMITIVE_RECORDER = false;

        /** @brief True for a type that declares itself a PrimitiveRecorder with K_PRIMITIVE_RECORDER. */
        template <class Port>
        inline constexpr bool K_IS_PRIMITIVE_RECORDER<
            Port, std::void_t<decltype(Port::K_PRIMITIVE_RECORDER)>> = Port::K_PRIMITIVE_RECORDER;

    } // namespace GraphPrimitives
} // namespace SushiBLAS
