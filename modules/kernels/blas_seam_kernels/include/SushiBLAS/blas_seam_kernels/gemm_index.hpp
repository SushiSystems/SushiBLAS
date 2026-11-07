/**************************************************************************/
/* gemm_index.hpp                                                         */
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
 * @file gemm_index.hpp
 * @brief Declares the group, staging, fold and fragment index math of the SYCL GEMM kernels, shared by the host and the device.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <SushiBLAS/blas_seam_kernels/extent_index.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace Indexing
        {
            /** @brief Holds a two-dimensional work-group shape, in work-items. */
            struct GroupShape
            {
                std::size_t rows = 1;   /**< Work-items down the row axis. */
                std::size_t cols = 1;   /**< Work-items across the column axis. */
            };

            /**
             * @brief Splits a work-item budget into the squarest power-of-two 2-D shape.
             * @return A shape whose product is at most budget and never zero.
             * @see modules/kernels/blas_seam_kernels/README.md
             */
            [[nodiscard]] constexpr GroupShape group_shape(std::size_t budget) noexcept
            {
                std::size_t side = 1;
                while ((side * 2) * (side * 2) <= budget)
                    side *= 2;

                std::size_t rows = side;
                if (rows * 2 * side <= budget)
                    rows *= 2;

                return GroupShape{rows, side};
            }

            /** @brief Returns the group count covering extent in steps of block, at least one. */
            [[nodiscard]] constexpr int64_t group_count(int64_t extent, int64_t block) noexcept
            {
                const int64_t count = ceil_div(extent, block);
                return count > 0 ? count : int64_t{1};
            }

            /** @brief Returns `group * block + local`: the first coordinate a work-item owns. */
            [[nodiscard]] constexpr int64_t block_origin(int64_t group, int64_t block,
                                                         int64_t local) noexcept
            {
                return group * block + local;
            }

            /** @brief Returns `origin + step_index * step`: a coordinate of a strided lattice. */
            [[nodiscard]] constexpr int64_t lattice_coord(int64_t origin, int64_t step_index,
                                                          int64_t step) noexcept
            {
                return origin + step_index * step;
            }

            /** @brief Holds a row and a column coordinate. */
            struct GridCoord
            {
                int64_t row = 0;    /**< Row coordinate. */
                int64_t col = 0;    /**< Column coordinate. */
            };

            /**
             * @brief Splits a staging index of a rows x cols block into `(row, col)`, walking down
             *        the columns when neighbouring rows are adjacent, else along the rows.
             */
            [[nodiscard]] constexpr GridCoord stage_coord(int64_t idx, int64_t rows, int64_t cols,
                                                          bool columns_contiguous) noexcept
            {
                return columns_contiguous ? GridCoord{idx % rows, idx / rows}
                                          : GridCoord{idx / cols, idx % cols};
            }

            /**
             * @brief Splits a vector-load index of a rows x cols block into the `(row, col)` of its
             *        first lane; one load takes `lanes` rows of a column or `lanes` columns of a row.
             * @pre The side the loads run along is a multiple of lanes.
             */
            [[nodiscard]] constexpr GridCoord stage_vector_coord(int64_t v, int64_t rows,
                                                                 int64_t cols, int64_t lanes,
                                                                 bool columns_contiguous) noexcept
            {
                const int64_t row_steps = columns_contiguous ? rows / lanes : rows;
                const int64_t col_steps = columns_contiguous ? cols : cols / lanes;
                return columns_contiguous ? GridCoord{(v % row_steps) * lanes, v / row_steps}
                                          : GridCoord{v / col_steps, (v % col_steps) * lanes};
            }

            /** @brief Returns whether a block of rows x cols at `(row0, col0)` lies wholly inside the matrix. */
            [[nodiscard]] constexpr bool block_is_whole(int64_t row0, int64_t col0, int64_t rows,
                                                        int64_t cols, int64_t matrix_rows,
                                                        int64_t matrix_cols) noexcept
            {
                return row0 + rows <= matrix_rows && col0 + cols <= matrix_cols;
            }

            /** @brief Returns the stride of the axis of an operand that is not contiguous, or 0 when neither is. */
            [[nodiscard]] constexpr int64_t lead_stride(int64_t row_stride,
                                                        int64_t col_stride) noexcept
            {
                return row_stride == 1 ? col_stride : (col_stride == 1 ? row_stride : 0);
            }

            /** @brief Returns the stride a fragment load steps by: the row stride when columns step by one, else the column stride. */
            [[nodiscard]] constexpr int64_t fragment_stride(int64_t row_stride,
                                                            int64_t col_stride) noexcept
            {
                return col_stride == 1 ? row_stride : col_stride;
            }

            /** @brief Returns whether a leading dimension and a batch stride are whole multiples of the given counts. */
            [[nodiscard]] constexpr bool strides_are_aligned(int64_t lead, int64_t lead_multiple,
                                                             int64_t batch_stride,
                                                             int64_t batch_multiple) noexcept
            {
                return lead > 0 && lead % lead_multiple == 0 && batch_stride % batch_multiple == 0;
            }

            /** @brief Returns whether every offset inside a rows x cols block of positive strides fits in 32 bits. */
            [[nodiscard]] constexpr bool block_fits_int32(int64_t row_stride, int64_t col_stride,
                                                          int64_t rows, int64_t cols) noexcept
            {
                const int64_t last = (rows - 1) * row_stride + (cols - 1) * col_stride;
                return row_stride > 0 && col_stride > 0 && last <= INT32_MAX;
            }

            /** @brief Returns the 32-bit offset of `(r, c)` inside a block whose offsets fit 32 bits. */
            [[nodiscard]] constexpr int32_t narrow_offset(int r, int c, int64_t row_stride,
                                                          int64_t col_stride) noexcept
            {
                return static_cast<int32_t>(r) * static_cast<int32_t>(row_stride) +
                       static_cast<int32_t>(c) * static_cast<int32_t>(col_stride);
            }

            /** @brief Holds the batch entry and the block of one group of the depth-split launch. */
            struct SplitGroup
            {
                int64_t batch = 0;  /**< Batch entry. */
                int64_t block = 0;  /**< Block of C inside that entry. */
            };

            /** @brief Splits a flat group id into its batch entry and block. */
            [[nodiscard]] constexpr SplitGroup split_group(int64_t group, int64_t blocks) noexcept
            {
                return SplitGroup{group / blocks, group % blocks};
            }

            /** @brief Holds the column lane and the depth lane of one work-item of the depth-split launch. */
            struct SplitLane
            {
                std::size_t column = 0; /**< Position along the columns of op(B). */
                std::size_t depth = 0;  /**< Position along the K extent. */
            };

            /** @brief Splits a lane id: neighbouring lanes take neighbouring columns, then K-steps. */
            [[nodiscard]] constexpr SplitLane split_lane(std::size_t lane,
                                                         std::size_t column_lanes) noexcept
            {
                return SplitLane{lane % column_lanes, lane / column_lanes};
            }

            /** @brief Returns the first row and column of C a depth-split work-item owns. */
            [[nodiscard]] constexpr GridCoord split_origin(int64_t block, int64_t n_blocks,
                                                           int64_t item_rows, int64_t block_cols,
                                                           std::size_t column_lane,
                                                           int64_t item_cols) noexcept
            {
                return GridCoord{(block / n_blocks) * item_rows,
                                 (block % n_blocks) * block_cols +
                                     static_cast<int64_t>(column_lane) * item_cols};
            }

            /** @brief Returns the lane whose partial the fold adds to lane `lane` at distance `off` depth lanes. */
            [[nodiscard]] constexpr std::size_t fold_partner(std::size_t lane, std::size_t off,
                                                             std::size_t column_lanes) noexcept
            {
                return lane + off * column_lanes;
            }

            /** @brief Returns `element * width + lane`: where a work-item keeps one partial in the fold tile. */
            [[nodiscard]] constexpr std::size_t partial_slot(std::size_t element, std::size_t width,
                                                             std::size_t lane) noexcept
            {
                return element * width + lane;
            }

            /**
             * @brief Returns the first row and column of C that sub-group `sub` of a work-group
             *        owns in the joint_matrix launch.
             * @param block Rows and columns of C one work-group owns.
             * @param sub_block Rows and columns of C one sub-group owns.
             * @param sub_groups Sub-groups of a work-group along each side.
             */
            [[nodiscard]] constexpr GridCoord sub_group_origin(int64_t group_row, int64_t group_col,
                                                               int64_t sub, int64_t block,
                                                               int64_t sub_block,
                                                               int64_t sub_groups) noexcept
            {
                return GridCoord{group_row * block + (sub / sub_groups) * sub_block,
                                 group_col * block + (sub % sub_groups) * sub_block};
            }

            /** @brief Returns whether fragment `f` of a sub-group that starts at `origin` begins inside `extent`. */
            [[nodiscard]] constexpr bool fragment_inside(int64_t origin, int64_t f,
                                                         int64_t fragment, int64_t extent) noexcept
            {
                return origin + f * fragment < extent;
            }

            /** @brief Splits an element index of a row-major fragment into its row and column. */
            [[nodiscard]] constexpr GridCoord fragment_element(int64_t e, int64_t fragment) noexcept
            {
                return GridCoord{e / fragment, e % fragment};
            }
        }
    }
}
