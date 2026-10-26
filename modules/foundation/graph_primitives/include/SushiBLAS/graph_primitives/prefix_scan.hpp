/**************************************************************************/
/* prefix_scan.hpp                                                        */
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
 * @file prefix_scan.hpp
 * @brief Declares prefix_scan(), an exclusive scan of uint32 counts into offsets.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>

#include <SushiRuntime/api/data/buffer.hpp>
#include <SushiRuntime/api/graph/recorder.hpp>
#include <SushiRuntime/api/vocabulary/late_count.hpp>
#include <SushiRuntime/core/result.hpp>

#include <SushiBLAS/graph_primitives/primitive_options.hpp>

namespace SushiBLAS
{
    namespace GraphPrimitives
    {
        /**
         * @brief Places the operands of prefix_scan() inside their buffers, in elements.
         * @see modules/foundation/graph_primitives/README.md
         */
        struct PrefixScanWindow
        {
            std::size_t counts_offset = 0;  /**< Element of counts the scan reads first. */
            std::size_t offsets_offset = 0; /**< Element of offsets that receives offset 0. */
            std::size_t total_offset = 0;   /**< Element of total that receives the sum. */
        };

        /**
         * @brief Records an exclusive scan of uint32 counts into running offsets.
         * @param counts  Holds at least n.capacity() elements.
         * @param offsets Holds at least n.capacity() + 1; offsets[live n] receives the sum.
         * @param total   Holds at least one element; total[0] receives the sum.
         * @param options Names and classes every node; none leaves them unnamed and Unstated.
         * @return Nothing, or an error the README lists; a run fails past n.capacity().
         */
        [[nodiscard]] SushiRuntime::Result<void>
        prefix_scan(SushiRuntime::Recorder& recorder,
                    const SushiRuntime::Buffer<std::uint32_t>& counts,
                    const SushiRuntime::Buffer<std::uint32_t>& offsets,
                    const SushiRuntime::Buffer<std::uint32_t>& total,
                    SushiRuntime::LateCount n, PrefixScanWindow window = {},
                    const std::optional<PrimitiveOptions>& options = std::nullopt);

    } // namespace GraphPrimitives
} // namespace SushiBLAS
