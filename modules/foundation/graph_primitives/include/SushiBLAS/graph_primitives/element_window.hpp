/**************************************************************************/
/* element_window.hpp                                                     */
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
 * @file element_window.hpp
 * @brief Declares the element interval of a buffer a primitive's node reads or writes.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <limits>
#include <string>

#include <SushiRuntime/api/data/buffer.hpp>
#include <SushiRuntime/api/vocabulary/dependencies.hpp>
#include <SushiRuntime/core/result.hpp>
#include <SushiRuntime/region/buffer_region.hpp>

#include <SushiBLAS/graph_primitives/invalid_graph.hpp>

namespace SushiBLAS
{
    namespace GraphPrimitives
    {
        namespace Detail
        {
            /**
             * @brief Names elements [offset, offset + count) of one buffer.
             * @see modules/foundation/graph_primitives/README.md
             */
            template <typename T>
            struct ElementSpan
            {
                const SushiRuntime::Buffer<T>* buffer = nullptr; /**< Buffer the span lies in. */
                std::size_t offset = 0;                          /**< First element. */
                std::size_t count = 0;                           /**< Elements spanned. */

                /** @brief Returns the address of the span's first element. */
                [[nodiscard]] T* data() const noexcept { return buffer->data() + offset; }
            };

            /** @brief Returns the span of count elements of buffer from offset. */
            template <typename T>
            [[nodiscard]] ElementSpan<T> elements(const SushiRuntime::Buffer<T>& buffer,
                                                  std::size_t offset, std::size_t count) noexcept
            {
                return ElementSpan<T>{&buffer, offset, count};
            }

            /**
             * @brief Checks that elements [offset, offset + count) lie inside a buffer.
             * @param primitive Name the message opens with.
             * @param operand   Name of the buffer argument the message names.
             * @return Nothing, or an InvalidGraph error naming the operand and the interval.
             */
            [[nodiscard]] inline SushiRuntime::Result<void>
            require_window(const char* primitive, const char* operand, std::size_t size,
                           std::size_t offset, std::size_t count)
            {
                if (offset <= size && count <= size - offset)
                    return {};

                const bool end_wraps = count > std::numeric_limits<std::size_t>::max() - offset;
                const std::string end = end_wraps ? std::string("past the largest size")
                                                  : std::to_string(offset + count);
                return invalid_graph(std::string(primitive) + ": the " + operand +
                                     " window [" + std::to_string(offset) + ", " + end +
                                     ") does not fit its buffer of " + std::to_string(size) +
                                     " elements.");
            }

            /** @brief Returns a bare region as the handle a dependency set collects. */
            [[nodiscard]] inline const SushiRuntime::BufferRegion&
            handle_of(const SushiRuntime::BufferRegion& region) noexcept
            {
                return region;
            }

            /** @brief Returns the buffer of a span as the handle a dependency set collects. */
            template <typename T>
            [[nodiscard]] const SushiRuntime::Buffer<T>&
            handle_of(const ElementSpan<T>& span) noexcept
            {
                return *span.buffer;
            }

            /** @brief Leaves the region of a bare region as it was named. */
            inline void narrow(SushiRuntime::BufferRegion&,
                               const SushiRuntime::BufferRegion&) noexcept
            {

            }

            /** @brief Narrows the whole-buffer region of a span's buffer to the span's bytes. */
            template <typename T>
            void narrow(SushiRuntime::BufferRegion& key, const ElementSpan<T>& span) noexcept
            {
                key.offset += span.offset * sizeof(T);
                key.length = span.count * sizeof(T);
            }

            /**
             * @brief Returns Reads or Writes over spans and bare regions, watching each buffer.
             * @tparam Set SushiRuntime::Reads or SushiRuntime::Writes.
             * @see modules/foundation/graph_primitives/README.md
             */
            template <class Set, class... Operands>
            [[nodiscard]] Set access_of(const Operands&... operands)
            {
                Set set(handle_of(operands)...);
                std::size_t slot = 0;
                (narrow(set.keys[slot++], operands), ...);
                return set;
            }

            /** @brief Returns the region a span names, keyed as its buffer is keyed. */
            template <typename T>
            [[nodiscard]] SushiRuntime::BufferRegion region_of(const ElementSpan<T>& span)
            {
                return access_of<SushiRuntime::Reads>(span).keys.front();
            }

        } // namespace Detail
    } // namespace GraphPrimitives
} // namespace SushiBLAS
