/**************************************************************************/
/* fft_line_index.hpp                                                     */
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
 * @file fft_line_index.hpp
 * @brief Declares the line layout, butterfly positions and spectrum mirroring of the batched FFT, shared by the host and the device.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Indexing
        {
            /**
             * @brief Holds how a set of strided lines sits in memory, in complex elements.
             * @note Line l is line l % per_group of group l / per_group.
             */
            struct FftLineLayout
            {
                    std::size_t lines = 0;          /**< Lines transformed together. */
                    std::size_t per_group = 1;      /**< Lines in one evenly strided group. */
                    std::size_t group_stride = 0;   /**< Distance between groups. */
                    std::size_t line_stride = 0;    /**< Distance between lines of a group. */
                    std::size_t element_stride = 1; /**< Distance between a line's elements. */

                    /** @brief Returns the complex offset of element @p k of line @p l. */
                    [[nodiscard]] constexpr std::size_t at(std::size_t l, std::size_t k) const noexcept
                    {
                        return (l / per_group) * group_stride + (l % per_group) * line_stride +
                               k * element_stride;
                    }
            };

            /** @brief Returns the layout of @p lines contiguous lines @p length apart. */
            [[nodiscard]] constexpr FftLineLayout contiguous_lines(std::size_t lines,
                                                                   std::size_t length) noexcept
            {
                FftLineLayout layout;
                layout.lines = lines;
                layout.per_group = lines == 0 ? 1 : lines;
                layout.line_stride = length;
                return layout;
            }

            /**
             * @brief Returns the layout of the columns of @p batch spectra of @p rows rows and
             *        @p width complex columns.
             */
            [[nodiscard]] constexpr FftLineLayout column_layout(std::size_t batch, std::size_t rows,
                                                                std::size_t width) noexcept
            {
                FftLineLayout layout;
                layout.lines = batch * width;
                layout.per_group = width;
                layout.group_stride = rows * width;
                layout.line_stride = 1;
                layout.element_stride = width;
                return layout;
            }

            /** @brief Holds one flat work index as a line and an element of it. */
            struct FftFlatCoord
            {
                std::size_t line = 0;    /**< Line index. */
                std::size_t element = 0; /**< Element in the line. */
            };

            /**
             * @brief Splits a flat index, element fastest, into `(line, element)`.
             * @param width Must be positive.
             */
            [[nodiscard]] constexpr FftFlatCoord fft_flat_coord(std::size_t flat,
                                                                std::size_t width) noexcept
            {
                return FftFlatCoord{flat / width, flat % width};
            }

            /** @brief Holds the positions one butterfly reads and writes in a line. */
            struct ButterflyPositions
            {
                std::size_t upper = 0;   /**< Element that gets the sum. */
                std::size_t lower = 0;   /**< Element that gets the difference. */
                std::size_t twiddle = 0; /**< Index of the root in the twiddle table. */
            };

            /**
             * @brief Returns the positions of the butterfly `k` of the block that starts at `start`.
             * @param half Half the span of the pass.
             * @param stride The line size divided by the span of the pass.
             */
            [[nodiscard]] constexpr ButterflyPositions butterfly_at(std::size_t start, std::size_t k,
                                                                    std::size_t half,
                                                                    std::size_t stride) noexcept
            {
                return ButterflyPositions{start + k, start + k + half, k * stride};
            }

            /**
             * @brief Returns the positions of the b-th butterfly of one pass, blocks in order.
             * @param span The span of the pass, a power of two of at least 2.
             * @param size The radix-2 line size.
             */
            [[nodiscard]] constexpr ButterflyPositions butterfly_number(std::size_t b,
                                                                        std::size_t span,
                                                                        std::size_t size) noexcept
            {
                const std::size_t half = span / 2;
                return butterfly_at((b / half) * span, b % half, half, size / span);
            }

            /** @brief Returns whether lane @p k of a padded Bluestein line lies past the data. */
            [[nodiscard]] constexpr bool is_padding_lane(std::size_t k, std::size_t length) noexcept
            {
                return k >= length;
            }

            /** @brief Returns the bin of a packed line that pairs with bin @p k: `(half - k) % half`. */
            [[nodiscard]] constexpr std::size_t mirror_bin(std::size_t k, std::size_t half) noexcept
            {
                return (half - k) % half;
            }

            /** @brief Holds where a full-length line element reads the half spectrum from. */
            struct HermitianSource
            {
                std::size_t bin = 0;     /**< Stored bin to read. */
                bool conjugated = false; /**< Whether the read value is conjugated. */
            };

            /**
             * @brief Returns the stored bin that supplies element @p k of a full-length line of a
             *        real signal, and whether it is conjugated.
             * @param width The stored bins, columns / 2 + 1.
             */
            [[nodiscard]] constexpr HermitianSource hermitian_source(std::size_t k, std::size_t width,
                                                                     std::size_t columns) noexcept
            {
                return k < width ? HermitianSource{k, false} : HermitianSource{columns - k, true};
            }
        }
    }
}
