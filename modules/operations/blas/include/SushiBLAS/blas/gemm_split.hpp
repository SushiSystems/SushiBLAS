/**************************************************************************/
/* gemm_split.hpp                                                         */
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
 * @file gemm_split.hpp
 * @brief Declares how a GEMM states that the runtime may cut it along the rows or the columns of C.
 * @author Mustafa Garip
 */

#pragma once

#include <memory>
#include <optional>
#include <cstddef>
#include <vector>
#include <cstdint>
#include <utility>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/blas/shape_check.hpp>
#include <SushiBLAS/blas_seam_kernels/sycl/level3/gemm.hpp>
#include <SushiBLAS/engine/operand_pointers.hpp>
#include <SushiBLAS/blas/gemm_bindings.hpp>
#include <SushiBLAS/blas/gemm_split_regions.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiRuntime/machine/node_split.hpp>

namespace SushiBLAS
{
    namespace Dispatch
    {
        /// @brief The operands of a GEMM the runtime binds a pointer for: A, B and C.
        inline constexpr std::size_t K_GEMM_OPERAND_COUNT = 3;

        /**
         * @brief Holds the sub-problem one part of a split GEMM computes: its extents and the element
         *        offsets of its operands from the whole call's.
         * @see modules/operations/blas/README.md
         */
        struct GemmSlice
        {
            int64_t m = 0;        /**< Rows of C the part computes. */
            int64_t n = 0;        /**< Columns of C the part computes. */
            int64_t a_offset = 0; /**< Elements from the base of A to the part's first element of A. */
            int64_t b_offset = 0; /**< Elements from the base of B to the part's first element of B. */
            int64_t c_offset = 0; /**< Elements from the base of C to the part's first element of C. */
        };

        /**
         * @brief Returns the axis a GEMM is cut along: the longer side of C, rows on a tie.
         * @see modules/operations/blas/README.md
         */
        [[nodiscard]] constexpr GemmSplitAxis gemm_split_axis(const GemmGeometry& geometry) noexcept
        {
            return geometry.m >= geometry.n ? GemmSplitAxis::Rows : GemmSplitAxis::Columns;
        }

        /**
         * @brief Returns the length of the axis, the extent a NodeSplit states.
         * @see modules/operations/blas/README.md
         */
        [[nodiscard]] constexpr int64_t gemm_split_extent(const GemmGeometry& geometry,
                                                          GemmSplitAxis axis) noexcept
        {
            return axis == GemmSplitAxis::Rows ? geometry.m : geometry.n;
        }

        /**
         * @brief Returns the sub-problem of the part that covers [begin, end) of the axis.
         *
         * A row of op(X) steps by the leading dimension in a row major X that is not transposed
         * or a column major X that is, and by one element otherwise; a column steps the other way.
         * @pre 0 <= begin <= end <= gemm_split_extent(geometry, axis).
         * @see modules/operations/blas/README.md
         */
        [[nodiscard]] constexpr GemmSlice
        slice_gemm(const GemmGeometry& geometry, Core::Layout layout, Core::Transpose trans_a,
                   Core::Transpose trans_b, GemmSplitAxis axis, int64_t begin, int64_t end) noexcept
        {
            const bool row_major = layout == Core::Layout::RowMajor;
            const bool a_rows_strided = row_major != Core::is_transposed(trans_a);
            const bool b_rows_strided = row_major != Core::is_transposed(trans_b);

            GemmSlice slice;
            slice.m = geometry.m;
            slice.n = geometry.n;
            if (axis == GemmSplitAxis::Rows)
            {
                slice.m = end - begin;
                slice.a_offset = begin * (a_rows_strided ? geometry.a.leading_dimension : 1);
                slice.c_offset = begin * (row_major ? geometry.c.leading_dimension : 1);
            }
            else
            {
                slice.n = end - begin;
                slice.b_offset = begin * (b_rows_strided ? 1 : geometry.b.leading_dimension);
                slice.c_offset = begin * (row_major ? 1 : geometry.c.leading_dimension);
            }
            return slice;
        }

        /**
         * @brief Wraps an epilogue so that a part's local row and column reach it as the row and
         *        column within the whole matrix.
         * @tparam Epilogue The functor the whole call applies.
         * @see modules/operations/blas/README.md
         */
        template<typename Epilogue>
        class OffsetEpilogue
        {
            public:
                /** @brief Holds the functor and the position of the part's first element in C. */
                OffsetEpilogue(Epilogue epilogue, int64_t row_offset, int64_t column_offset)
                    : epilogue_(std::move(epilogue)), row_offset_(row_offset),
                      column_offset_(column_offset)
                {

                }

                /** @brief Applies the functor at the position of the element in the whole matrix. */
                template<typename T>
                auto operator()(T value, int64_t row, int64_t column) const
                {
                    return epilogue_(value, row + row_offset_, column + column_offset_);
                }

            private:
                /// @brief The functor the whole call applies.
                Epilogue epilogue_;

                /// @brief Rows of C before the part.
                int64_t row_offset_;

                /// @brief Columns of C before the part.
                int64_t column_offset_;
        };

        /**
         * @brief Holds the three operand pointers of a GEMM, as one part's kernel reads them.
         * @tparam T The element type of A, B and C.
         * @see modules/operations/blas/README.md
         */
        template<typename T>
        struct GemmOperands
        {
            const T* a = nullptr; /**< The first element of A. */
            const T* b = nullptr; /**< The first element of B. */
            T* c = nullptr;       /**< The first element of C. */
        };

        /**
         * @brief Returns the first elements of A, B and C in the bound table, or the recorded ones.
         * @param bound The base address of each allocation the task bound.
         * @param bindings Where A, B and C sit in bound; empty for a call that bound nothing.
         * @param recorded The pointers the call recorded; read only when bindings is empty.
         * @pre When bindings holds a value, bound holds the three allocations.
         * @see modules/operations/blas/README.md
         */
        template<typename T>
        [[nodiscard]] GemmOperands<T>
        bound_gemm_operands(const OperandPointers& bound,
                            const std::optional<GemmBindings>& bindings,
                            const GemmOperands<T>& recorded) noexcept
        {
            if (!bindings.has_value())
                return recorded;
            return GemmOperands<T>{bound.as<const T>(bindings->a), bound.as<const T>(bindings->b),
                                   bound.as<T>(bindings->c)};
        }

        /**
         * @brief Builds the split of a GEMM: gemm_part_regions() and the SYCL GEMM of each part.
         * @param a,b,c The first elements of A, B and C as recorded; null for a bound call.
         * @param operand_regions Where A, B and C start in their allocations; C is read when beta is not zero.
         * @param bindings Where the bound table holds A, B and C; empty for a call that bound none.
         * @return A split along gemm_split_axis(geometry).
         * @see modules/operations/blas/README.md
         */
        template<typename T, typename Epilogue>
        [[nodiscard]] std::shared_ptr<const SushiRuntime::NodeSplit>
        make_gemm_split(const GemmGeometry& geometry, Core::Layout layout, Core::Transpose trans_a,
                        Core::Transpose trans_b, T alpha, const T* a, const T* b, T beta, T* c,
                        const Epilogue& epilogue, const GemmOperandRegions& operand_regions,
                        const std::optional<GemmBindings>& bindings = std::nullopt)
        {
            const GemmSplitAxis axis = gemm_split_axis(geometry);
            const GemmOperands<T> recorded{a, b, c};
            auto split = std::make_shared<SushiRuntime::NodeSplit>();
            split->axis = static_cast<std::uint32_t>(axis);
            split->extent = static_cast<std::uint64_t>(gemm_split_extent(geometry, axis));
            const bool reads_c = beta != T(0);
            split->regions = [=](const SushiRuntime::SplitPart& part)
            {
                return gemm_part_regions(geometry, layout, trans_a, trans_b, reads_c, axis,
                                         static_cast<int64_t>(part.begin),
                                         static_cast<int64_t>(part.end), sizeof(T),
                                         operand_regions);
            };
            split->slice = [=](const SushiRuntime::SplitPart& part, sycl::queue& queue,
                               SushiRuntime::span<void* const> table)
            {
                const int64_t begin = static_cast<int64_t>(part.begin);
                const GemmSlice slice = slice_gemm(geometry, layout, trans_a, trans_b, axis, begin,
                                                   static_cast<int64_t>(part.end));
                const int64_t row_offset = axis == GemmSplitAxis::Rows ? begin : 0;
                const int64_t column_offset = axis == GemmSplitAxis::Columns ? begin : 0;
                const GemmOperands<T> operands = bound_gemm_operands<T>(
                    OperandPointers(table.data(), table.size()), bindings, recorded);
                Kernels::SYCL::gemm<T>(queue, layout, trans_a, trans_b, slice.m, slice.n,
                                       geometry.k, alpha, operands.a + slice.a_offset,
                                       geometry.a.leading_dimension, geometry.stride_a,
                                       operands.b + slice.b_offset, geometry.b.leading_dimension,
                                       geometry.stride_b, beta, operands.c + slice.c_offset,
                                       geometry.c.leading_dimension, geometry.stride_c,
                                       geometry.batch_size, std::vector<sycl::event>{},
                                       OffsetEpilogue<Epilogue>(epilogue, row_offset,
                                                                column_offset)).wait();
            };
            return split;
        }
    } // namespace Dispatch
} // namespace SushiBLAS
