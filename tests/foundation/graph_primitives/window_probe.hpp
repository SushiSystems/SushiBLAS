/**************************************************************************/
/* window_probe.hpp                                                       */
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
 * @file window_probe.hpp
 * @brief Declares the probe that reads whether a node over a buffer window is ordered.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>

#include <SushiRuntime/api/data/buffer.hpp>
#include <SushiRuntime/api/graph/graph.hpp>
#include <SushiRuntime/api/graph/plan_view.hpp>
#include <SushiRuntime/api/vocabulary/dependencies.hpp>
#include <SushiRuntime/region/buffer_region.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Holds the probe the buffer window tests of the primitives share. */
        namespace WindowProbe
        {
            /** @brief Returns elements [offset, offset + count) of buffer as a region. */
            template <typename T>
            [[nodiscard]] SushiRuntime::BufferRegion
            region_of(const SushiRuntime::Buffer<T>& buffer, std::size_t offset,
                      std::size_t count)
            {
                SushiRuntime::BufferRegion region = SushiRuntime::Reads(buffer).keys.front();
                region.offset += offset * sizeof(T);
                region.length = count * sizeof(T);
                return region;
            }

            /**
             * @brief Adds a node that writes region and returns how many nodes it waits for.
             * @pre The graph holds the nodes the probe is ordered against, and no probe yet.
             */
            [[nodiscard]] inline std::size_t
            predecessors_of_write(SushiRuntime::Graph& graph,
                                  const SushiRuntime::BufferRegion& region)
            {
                graph.add(std::size_t{1}, SushiRuntime::Reads(), SushiRuntime::Writes(region),
                          [](std::size_t)
                          {

                          });
                return graph.plan().value().nodes.back().predecessor_count;
            }
        } // namespace WindowProbe
    } // namespace Test
} // namespace SushiBLAS
