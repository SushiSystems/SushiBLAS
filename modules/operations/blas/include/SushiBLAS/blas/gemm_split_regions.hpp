/**************************************************************************/
/* gemm_split_regions.hpp                                                 */
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
 * @file gemm_split_regions.hpp
 * @brief Declares the byte regions one part of a split GEMM reads and writes.
 * @author Mustafa Garip
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/blas/shape_check.hpp>
#include <SushiRuntime/machine/split_regions.hpp>
#include <SushiRuntime/region/buffer_region.hpp>

namespace SushiBLAS
{
    namespace Dispatch
    {
        /**
         * @brief Names the axis of C a GEMM is cut along, as the numbers NodeSplit::axis carries.
         * @see modules/operations/blas/README.md
         */
        enum class GemmSplitAxis : std::uint32_t
        {
            Rows = 0,   /**< Parts hold consecutive rows of C and of op(A). */
            Columns = 1 /**< Parts hold consecutive columns of C and of op(B). */
        };

        /**
         * @brief Holds where A, B and C start and what every part reads whole.
         * @see modules/operations/blas/README.md
         */
        struct GemmOperandRegions
        {
            SushiRuntime::BufferRegion a; /**< The handle of A and the byte offset of its first element. */
            SushiRuntime::BufferRegion b; /**< The handle of B and the byte offset of its first element. */
            SushiRuntime::BufferRegion c; /**< The handle of C and the byte offset of its first element. */
            std::vector<SushiRuntime::BufferRegion> whole_reads; /**< Regions every part reads whole, such as an epilogue's operands. */
        };

        namespace Detail
        {
            /**
             * @brief Holds the rows [row_begin, row_end) and columns [column_begin, column_end) of
             *        one stored matrix, and how the matrix lies in its allocation.
             */
            struct StoredBlock
            {
                SushiRuntime::BufferRegion start; /**< The handle and the byte offset of the first element. */
                bool row_major = true;            /**< Whether a row is contiguous. */
                int64_t leading_dimension = 0;    /**< Elements between two rows (row-major) or two columns. */
                int64_t stride = 0;               /**< Elements between two matrices of the batch. */
                int64_t batch_size = 1;           /**< Matrices of the batch. */
                int64_t row_begin = 0;            /**< First row of the block. */
                int64_t row_end = 0;              /**< One past the last row. */
                int64_t column_begin = 0;         /**< First column of the block. */
                int64_t column_end = 0;           /**< One past the last column. */
            };

            /** @brief Appends one range per contiguous run of block, for each matrix of the batch. */
            inline void append_block(std::vector<SushiRuntime::BufferRegion>& out,
                                     const StoredBlock& block, std::size_t element_bytes)
            {
                const int64_t line_begin = block.row_major ? block.row_begin : block.column_begin;
                const int64_t line_end = block.row_major ? block.row_end : block.column_end;
                const int64_t position_begin = block.row_major ? block.column_begin : block.row_begin;
                const int64_t position_end = block.row_major ? block.column_end : block.row_end;
                if (line_end <= line_begin || position_end <= position_begin)
                    return;
                const bool whole_lines = position_begin == 0 && position_end == block.leading_dimension;
                const auto append = [&out, &block, element_bytes](int64_t first_element, int64_t elements)
                {
                    out.emplace_back(block.start.base,
                                     block.start.offset + static_cast<std::size_t>(first_element) * element_bytes,
                                     static_cast<std::size_t>(elements) * element_bytes);
                };
                for (int64_t matrix = 0; matrix < block.batch_size; ++matrix)
                {
                    const int64_t origin = matrix * block.stride;
                    if (whole_lines)
                    {
                        append(origin + line_begin * block.leading_dimension,
                               (line_end - line_begin) * block.leading_dimension);
                        continue;
                    }
                    for (int64_t line = line_begin; line < line_end; ++line)
                        append(origin + line * block.leading_dimension + position_begin,
                               position_end - position_begin);
                }
            }

            /** @brief Returns one past the last byte of region; a K_WHOLE region has no end. */
            [[nodiscard]] inline std::size_t end_of(const SushiRuntime::BufferRegion& region) noexcept
            {
                if (region.length == SushiRuntime::BufferRegion::K_WHOLE)
                    return SushiRuntime::BufferRegion::K_WHOLE;
                return region.offset + region.length;
            }

            /** @brief Sorts regions by handle and offset and merges the ones that touch or overlap. */
            inline void coalesce(std::vector<SushiRuntime::BufferRegion>& regions)
            {
                std::sort(regions.begin(), regions.end(),
                          [](const SushiRuntime::BufferRegion& left, const SushiRuntime::BufferRegion& right)
                          {
                              if (left.base != right.base)
                                  return left.base < right.base;
                              return left.offset < right.offset;
                          });
                std::vector<SushiRuntime::BufferRegion> merged;
                merged.reserve(regions.size());
                for (const SushiRuntime::BufferRegion& region : regions)
                {
                    if (merged.empty() || merged.back().base != region.base ||
                        end_of(merged.back()) < region.offset)
                    {
                        merged.push_back(region);
                        continue;
                    }
                    SushiRuntime::BufferRegion& last = merged.back();
                    const std::size_t end = std::max(end_of(last), end_of(region));
                    last.length = end == SushiRuntime::BufferRegion::K_WHOLE
                                      ? SushiRuntime::BufferRegion::K_WHOLE
                                      : end - last.offset;
                }
                regions = std::move(merged);
            }
        } // namespace Detail

        /**
         * @brief Returns the bytes of A, B and C the part [begin, end) of the axis reads and writes.
         *
         * A row part reads its rows of op(A) and all of B, a column part all of A and its columns
         * of op(B); each writes its block of C, read too when reads_c holds, in every matrix.
         * @pre 0 <= begin <= end <= gemm_split_extent(geometry, axis).
         * @see modules/operations/blas/README.md
         */
        [[nodiscard]] inline SushiRuntime::SplitRegions
        gemm_part_regions(const GemmGeometry& geometry, Core::Layout layout, Core::Transpose trans_a,
                          Core::Transpose trans_b, bool reads_c, GemmSplitAxis axis, int64_t begin,
                          int64_t end, std::size_t element_bytes, const GemmOperandRegions& operands)
        {
            const bool row_major = layout == Core::Layout::RowMajor;
            const bool rows = axis == GemmSplitAxis::Rows;
            const bool a_swapped = Core::is_transposed(trans_a);
            const bool b_swapped = Core::is_transposed(trans_b);

            // Stored A is m x k, or k x m when transposed; a row part keeps rows [begin, end) of op(A).
            Detail::StoredBlock a{operands.a, row_major, geometry.a.leading_dimension, geometry.stride_a,
                                  geometry.batch_size, 0, a_swapped ? geometry.k : geometry.m,
                                  0, a_swapped ? geometry.m : geometry.k};
            if (rows && !a_swapped)
            {
                a.row_begin = begin;
                a.row_end = end;
            }
            else if (rows)
            {
                a.column_begin = begin;
                a.column_end = end;
            }

            // Stored B is k x n, or n x k when transposed; a column part keeps columns [begin, end) of op(B).
            Detail::StoredBlock b{operands.b, row_major, geometry.b.leading_dimension, geometry.stride_b,
                                  geometry.batch_size, 0, b_swapped ? geometry.n : geometry.k,
                                  0, b_swapped ? geometry.k : geometry.n};
            if (!rows && !b_swapped)
            {
                b.column_begin = begin;
                b.column_end = end;
            }
            else if (!rows)
            {
                b.row_begin = begin;
                b.row_end = end;
            }

            const Detail::StoredBlock c{operands.c, row_major, geometry.c.leading_dimension,
                                        geometry.stride_c, geometry.batch_size,
                                        rows ? begin : 0, rows ? end : geometry.m,
                                        rows ? 0 : begin, rows ? geometry.n : end};

            SushiRuntime::SplitRegions regions;
            Detail::append_block(regions.reads, a, element_bytes);
            Detail::append_block(regions.reads, b, element_bytes);
            if (reads_c)
                Detail::append_block(regions.reads, c, element_bytes);
            regions.reads.insert(regions.reads.end(), operands.whole_reads.begin(),
                                 operands.whole_reads.end());
            Detail::append_block(regions.writes, c, element_bytes);
            Detail::coalesce(regions.reads);
            Detail::coalesce(regions.writes);
            return regions;
        }
    } // namespace Dispatch
} // namespace SushiBLAS
