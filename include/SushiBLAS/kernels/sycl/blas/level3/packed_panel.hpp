/**************************************************************************/
/* packed_panel.hpp                                                       */
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
 * @file packed_panel.hpp
 * @brief Declares pack_panel, which copies a block of op(X) into unit-stride micro-panels.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/core/traits.hpp>
#include <SushiBLAS/support/accumulator.hpp>
#include <SushiBLAS/support/complex_arithmetic.hpp>
#include <SushiBLAS/kernels/blas/operand_access.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            namespace Detail
            {
                /**
                 * @brief Names the axis of op(X) whose lines a panel groups into micro-panels.
                 * @see include/SushiBLAS/kernels/README.md
                 */
                enum class PanelLines : std::uint8_t
                {
                    Rows,   /**< A line is a row and a term a column: the first factor. */
                    Columns /**< A line is a column and a term a row: the second factor. */
                };

                /**
                 * @brief Returns whether neighbouring lines of one term are adjacent in memory.
                 * @return false for an access brick that does not say; the answer steers the
                 *         order of the copy, never its result.
                 */
                template<typename Access>
                [[nodiscard]] constexpr bool
                has_adjacent_lines([[maybe_unused]] const Access& access,
                                   [[maybe_unused]] PanelLines lines) noexcept
                {
                    return false;
                }

                /** @brief Returns whether neighbouring lines of one term are one element apart. */
                [[nodiscard]] constexpr bool has_adjacent_lines(const OperandAccess& access,
                                                                PanelLines lines) noexcept
                {
                    return ((lines == PanelLines::Rows) ? access.row : access.col) == 1;
                }

                /**
                 * @brief Returns the element of op(X) at one line and one term of a panel.
                 * @tparam Access Brick with `T at(const T*, int64_t, int64_t) const`.
                 */
                template<PanelLines Lines, typename T, typename Access>
                [[nodiscard]] inline T panel_element(const T* x, const Access& access,
                                                     int64_t line, int64_t term) noexcept
                {
                    if constexpr (Lines == PanelLines::Rows)
                        return access.at(x, line, term);
                    else
                        return access.at(x, term, line);
                }

                /**
                 * @brief Writes one packed element: its real part, and for a complex W its
                 *        imaginary part Width reals further on.
                 * @param slice The first real of the term the element belongs to.
                 */
                template<typename W, int Width>
                inline void store_packed(RealType<W>* slice, int64_t line, W value) noexcept
                {
                    if constexpr (Core::is_complex_v<W>)
                    {
                        slice[line] = value.real();
                        slice[Width + line] = value.imag();
                    }
                    else
                    {
                        slice[line] = value;
                    }
                }

                /**
                 * @brief Packs count lines of op(X) over depth terms into micro-panels of Width
                 *        lines.
                 * @tparam Width The register rows for the first factor, the columns for the second.
                 * @tparam Access Brick with `T at(const T*, int64_t, int64_t) const`.
                 * @param panel Receives ceil(count / Width) micro-panels; lines past count are 0.
                 * @see include/SushiBLAS/kernels/README.md
                 */
                template<typename T, int Width, PanelLines Lines, typename Access>
                inline void pack_panel(const T* x, const Access& access, int64_t first,
                                       int64_t count, int64_t term_first, int64_t depth,
                                       RealType<Accumulator<T>>* panel) noexcept
                {
                    using W = Accumulator<T>;
                    using R = RealType<W>;
                    constexpr int64_t parts = Core::is_complex_v<W> ? 2 : 1;
                    constexpr int64_t slice_reals = parts * Width;

                    const int64_t micro_reals = depth * slice_reals;
                    const int64_t groups = (count + Width - 1) / Width;

                    // The inner loop of each arm walks the axis the source stores contiguously.
                    if (has_adjacent_lines(access, Lines))
                    {
                        for (int64_t l = 0; l < depth; ++l)
                        {
                            const int64_t term = term_first + l;
                            for (int64_t g = 0; g < groups; ++g)
                            {
                                const int64_t line0 = first + g * Width;
                                const int64_t left = count - g * Width;
                                const int64_t live = (left < Width) ? left : Width;
                                R* slice = panel + g * micro_reals + l * slice_reals;
                                for (int64_t r = 0; r < live; ++r)
                                    store_packed<W, Width>(
                                        slice, r,
                                        static_cast<W>(panel_element<Lines>(x, access, line0 + r,
                                                                            term)));
                            }
                        }
                    }
                    else
                    {
                        for (int64_t g = 0; g < groups; ++g)
                        {
                            const int64_t line0 = first + g * Width;
                            const int64_t left = count - g * Width;
                            const int64_t live = (left < Width) ? left : Width;
                            R* micro = panel + g * micro_reals;
                            for (int64_t r = 0; r < live; ++r)
                                for (int64_t l = 0; l < depth; ++l)
                                    store_packed<W, Width>(
                                        micro + l * slice_reals, r,
                                        static_cast<W>(panel_element<Lines>(x, access, line0 + r,
                                                                            term_first + l)));
                        }
                    }

                    // Only the last micro-panel can be short; its missing lines are zeroed here.
                    const int64_t tail = count - (groups - 1) * Width;
                    if (groups > 0 && tail < Width)
                    {
                        R* micro = panel + (groups - 1) * micro_reals;
                        for (int64_t l = 0; l < depth; ++l)
                            for (int64_t r = tail; r < Width; ++r)
                                store_packed<W, Width>(micro + l * slice_reals, r, W(0));
                    }
                }
            } // namespace Detail
        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
