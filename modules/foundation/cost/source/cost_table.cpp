/**************************************************************************/
/* cost_table.cpp                                                         */
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
 * @file cost_table.cpp
 * @brief Defines the cost rows of every operation and the rules that turn a shape into a Cost.
 * @author Mustafa Garip
 * @see modules/foundation/cost/README.md
 */

#include <SushiBLAS/cost/cost_table.hpp>

#include <array>
#include <limits>
#include <variant>
#include <algorithm>

namespace SushiBLAS
{
    using namespace SushiRuntime::Literals;

    namespace
    {
        /** @brief Builds the row of an operation that states a cost, with its three weights. */
        constexpr CostRow costed(SushiRuntime::OperationID id, std::uint32_t per_element,
                                 std::uint32_t per_combine, std::uint32_t per_output) noexcept
        {
            return CostRow{id, per_element, per_combine, per_output, UncostedReason::None};
        }

        /** @brief Builds the row of an operation that states no cost, for @p reason. */
        constexpr CostRow uncosted(SushiRuntime::OperationID id, UncostedReason reason) noexcept
        {
            return CostRow{id, 0, 0, 0, reason};
        }

        /** @brief Returns the rows ordered by id with an insertion sort, which C++17 can run at compile time. */
        template<std::size_t N>
        constexpr std::array<CostRow, N> sorted_by_id(std::array<CostRow, N> rows) noexcept
        {
            for (std::size_t i = 1; i < N; ++i)
            {
                const CostRow key = rows[i];
                std::size_t j = i;
                while (j > 0 && key.id < rows[j - 1].id)
                {
                    rows[j] = rows[j - 1];
                    --j;
                }
                rows[j] = key;
            }
            return rows;
        }

        /// @brief The rows in the order of their names.
        constexpr std::array K_NAMED_ROWS{
        costed("blas.lvl1.asum"_op, 1, 1, 0),
        costed("blas.lvl1.axpy"_op, 2, 0, 0),
        costed("blas.lvl1.copy"_op, 0, 0, 0),
        costed("blas.lvl1.dot"_op, 1, 1, 0),
        costed("blas.lvl1.dotc"_op, 1, 1, 0),
        costed("blas.lvl1.dotu"_op, 1, 1, 0),
        costed("blas.lvl1.iamax"_op, 1, 1, 0),
        costed("blas.lvl1.nrm2"_op, 1, 1, 1),
        uncosted("blas.lvl1.rot"_op, UncostedReason::Unaudited),
        uncosted("blas.lvl1.rotg"_op, UncostedReason::Unaudited),
        uncosted("blas.lvl1.rotm"_op, UncostedReason::Unaudited),
        uncosted("blas.lvl1.rotmg"_op, UncostedReason::Unaudited),
        costed("blas.lvl1.scal"_op, 1, 0, 0),
        costed("blas.lvl1.swap"_op, 0, 0, 0),
        uncosted("blas.lvl2.gbmv"_op, UncostedReason::Unaudited),
        uncosted("blas.lvl2.gemv"_op, UncostedReason::Unaudited),
        uncosted("blas.lvl2.ger"_op, UncostedReason::Unaudited),
        uncosted("blas.lvl2.gerc"_op, UncostedReason::Unaudited),
        uncosted("blas.lvl2.geru"_op, UncostedReason::Unaudited),
        uncosted("blas.lvl2.hbmv"_op, UncostedReason::Unaudited),
        uncosted("blas.lvl2.hemv"_op, UncostedReason::Unaudited),
        uncosted("blas.lvl2.her"_op, UncostedReason::Unaudited),
        uncosted("blas.lvl2.her2"_op, UncostedReason::Unaudited),
        uncosted("blas.lvl2.hpmv"_op, UncostedReason::Unaudited),
        uncosted("blas.lvl2.hpr"_op, UncostedReason::Unaudited),
        uncosted("blas.lvl2.hpr2"_op, UncostedReason::Unaudited),
        uncosted("blas.lvl2.sbmv"_op, UncostedReason::Unaudited),
        uncosted("blas.lvl2.spmv"_op, UncostedReason::Unaudited),
        uncosted("blas.lvl2.spr"_op, UncostedReason::Unaudited),
        uncosted("blas.lvl2.spr2"_op, UncostedReason::Unaudited),
        uncosted("blas.lvl2.symv"_op, UncostedReason::Unaudited),
        uncosted("blas.lvl2.syr"_op, UncostedReason::Unaudited),
        uncosted("blas.lvl2.syr2"_op, UncostedReason::Unaudited),
        uncosted("blas.lvl2.tbmv"_op, UncostedReason::Unaudited),
        uncosted("blas.lvl2.tbsv"_op, UncostedReason::Unaudited),
        uncosted("blas.lvl2.tpmv"_op, UncostedReason::Unaudited),
        uncosted("blas.lvl2.tpsv"_op, UncostedReason::Unaudited),
        uncosted("blas.lvl2.trmv"_op, UncostedReason::Unaudited),
        uncosted("blas.lvl2.trsv"_op, UncostedReason::Unaudited),
        costed("blas.lvl3.gemm"_op, 0, 0, 0),
        uncosted("blas.lvl3.hemm"_op, UncostedReason::Unaudited),
        uncosted("blas.lvl3.her2k"_op, UncostedReason::Unaudited),
        uncosted("blas.lvl3.herk"_op, UncostedReason::Unaudited),
        uncosted("blas.lvl3.symm"_op, UncostedReason::Unaudited),
        uncosted("blas.lvl3.syr2k"_op, UncostedReason::Unaudited),
        uncosted("blas.lvl3.syrk"_op, UncostedReason::Unaudited),
        uncosted("blas.lvl3.trsm"_op, UncostedReason::Unaudited),
        uncosted("blend.blend_transforms"_op, UncostedReason::Unaudited),
        uncosted("index.for_each"_op, UncostedReason::DataDependent),
        uncosted("linalg.gmres"_op, UncostedReason::Unaudited),
        uncosted("linalg.lu_factor"_op, UncostedReason::Unaudited),
        uncosted("linalg.lu_solve"_op, UncostedReason::Unaudited),
        uncosted("logic.all"_op, UncostedReason::Unaudited),
        costed("logic.and"_op, 1, 0, 0),
        uncosted("logic.any"_op, UncostedReason::Unaudited),
        costed("logic.equal"_op, 1, 0, 0),
        costed("logic.greater"_op, 1, 0, 0),
        costed("logic.greater_equal"_op, 1, 0, 0),
        costed("logic.less"_op, 1, 0, 0),
        costed("logic.less_equal"_op, 1, 0, 0),
        costed("logic.not"_op, 1, 0, 0),
        costed("logic.not_equal"_op, 1, 0, 0),
        costed("logic.or"_op, 1, 0, 0),
        costed("logic.where"_op, 1, 0, 0),
        costed("logic.xor"_op, 1, 0, 0),
        costed("math.argmax"_op, 0, 1, 0),
        costed("math.argmin"_op, 0, 1, 0),
        costed("math.asum"_op, 1, 1, 0),
        costed("math.count_nonfinite"_op, 1, 1, 0),
        costed("math.ew.abs"_op, 1, 0, 0),
        costed("math.ew.acos"_op, 1, 0, 0),
        costed("math.ew.acosh"_op, 1, 0, 0),
        costed("math.ew.add"_op, 1, 0, 0),
        costed("math.ew.asin"_op, 1, 0, 0),
        costed("math.ew.asinh"_op, 1, 0, 0),
        costed("math.ew.atan"_op, 1, 0, 0),
        costed("math.ew.atanh"_op, 1, 0, 0),
        costed("math.ew.ceil"_op, 1, 0, 0),
        costed("math.ew.clamp"_op, 2, 0, 0),
        costed("math.ew.convert"_op, 0, 0, 0),
        costed("math.ew.cos"_op, 1, 0, 0),
        costed("math.ew.cosh"_op, 1, 0, 0),
        costed("math.ew.div"_op, 1, 0, 0),
        costed("math.ew.exp"_op, 1, 0, 0),
        costed("math.ew.floor"_op, 1, 0, 0),
        costed("math.ew.fmod"_op, 1, 0, 0),
        costed("math.ew.log"_op, 1, 0, 0),
        costed("math.ew.max"_op, 1, 0, 0),
        costed("math.ew.min"_op, 1, 0, 0),
        costed("math.ew.mul"_op, 1, 0, 0),
        costed("math.ew.neg"_op, 1, 0, 0),
        uncosted("math.ew.ordered_sum"_op, UncostedReason::Unaudited),
        costed("math.ew.pow"_op, 1, 0, 0),
        costed("math.ew.reciprocal"_op, 1, 0, 0),
        costed("math.ew.remainder"_op, 1, 0, 0),
        costed("math.ew.round"_op, 1, 0, 0),
        costed("math.ew.sin"_op, 1, 0, 0),
        costed("math.ew.sinh"_op, 1, 0, 0),
        costed("math.ew.sqrt"_op, 1, 0, 0),
        costed("math.ew.square"_op, 1, 0, 0),
        costed("math.ew.sub"_op, 1, 0, 0),
        costed("math.ew.tan"_op, 1, 0, 0),
        uncosted("math.logsumexp"_op, UncostedReason::Unaudited),
        costed("math.max"_op, 0, 1, 0),
        uncosted("math.max_along"_op, UncostedReason::Unaudited),
        costed("math.mean"_op, 0, 1, 1),
        uncosted("math.mean_spatial"_op, UncostedReason::Unaudited),
        costed("math.min"_op, 0, 1, 0),
        uncosted("math.min_along"_op, UncostedReason::Unaudited),
        costed("math.nonlinear.elu"_op, 4, 0, 0),
        costed("math.nonlinear.elu_backward"_op, 4, 0, 0),
        costed("math.nonlinear.gelu"_op, 9, 0, 0),
        costed("math.nonlinear.gelu_backward"_op, 19, 0, 0),
        costed("math.nonlinear.leaky_relu"_op, 2, 0, 0),
        costed("math.nonlinear.leaky_relu_backward"_op, 2, 0, 0),
        costed("math.nonlinear.relu"_op, 1, 0, 0),
        costed("math.nonlinear.relu_backward"_op, 2, 0, 0),
        costed("math.nonlinear.sigmoid"_op, 4, 0, 0),
        costed("math.nonlinear.sigmoid_backward"_op, 3, 0, 0),
        costed("math.nonlinear.silu"_op, 4, 0, 0),
        costed("math.nonlinear.silu_backward"_op, 9, 0, 0),
        costed("math.nonlinear.softplus"_op, 6, 0, 0),
        costed("math.nonlinear.softplus_backward"_op, 5, 0, 0),
        costed("math.nonlinear.tanh"_op, 1, 0, 0),
        costed("math.nonlinear.tanh_backward"_op, 3, 0, 0),
        costed("math.nrm2"_op, 1, 1, 1),
        costed("math.prod"_op, 0, 1, 0),
        uncosted("math.std"_op, UncostedReason::Unaudited),
        costed("math.sum"_op, 0, 1, 0),
        uncosted("math.sum_along"_op, UncostedReason::Unaudited),
        uncosted("math.var"_op, UncostedReason::Unaudited),
        uncosted("random.bernoulli"_op, UncostedReason::Unaudited),
        uncosted("random.constant"_op, UncostedReason::Unaudited),
        uncosted("random.discrete_uniform"_op, UncostedReason::Unaudited),
        uncosted("random.exponential"_op, UncostedReason::Unaudited),
        uncosted("random.he_normal"_op, UncostedReason::Unaudited),
        uncosted("random.he_uniform"_op, UncostedReason::Unaudited),
        uncosted("random.log_normal"_op, UncostedReason::Unaudited),
        uncosted("random.normal"_op, UncostedReason::Unaudited),
        uncosted("random.orthogonal"_op, UncostedReason::Unaudited),
        uncosted("random.poisson"_op, UncostedReason::Unaudited),
        uncosted("random.shuffle"_op, UncostedReason::Unaudited),
        uncosted("random.truncated_normal"_op, UncostedReason::Unaudited),
        uncosted("random.uniform"_op, UncostedReason::Unaudited),
        uncosted("random.xavier_normal"_op, UncostedReason::Unaudited),
        uncosted("random.xavier_uniform"_op, UncostedReason::Unaudited),
        uncosted("signal.convolve_1d"_op, UncostedReason::Unaudited),
        costed("signal.fft"_op, 0, 0, 0),
        costed("signal.inverse_fft"_op, 0, 0, 0),
        uncosted("sparse.axpyi"_op, UncostedReason::Unaudited),
        uncosted("sparse.dotci"_op, UncostedReason::Unaudited),
        uncosted("sparse.doti"_op, UncostedReason::Unaudited),
        uncosted("sparse.dotui"_op, UncostedReason::Unaudited),
        uncosted("sparse.gthr"_op, UncostedReason::Unaudited),
        uncosted("sparse.gthrz"_op, UncostedReason::Unaudited),
        uncosted("sparse.roti"_op, UncostedReason::Unaudited),
        uncosted("sparse.sctr"_op, UncostedReason::Unaudited),
        uncosted("sparse.sphemv"_op, UncostedReason::Unaudited),
        uncosted("sparse.spmm"_op, UncostedReason::Unaudited),
        uncosted("sparse.spmv"_op, UncostedReason::Unaudited),
        uncosted("sparse.spsymv"_op, UncostedReason::Unaudited),
        uncosted("sparse.sptrsv"_op, UncostedReason::Unaudited),
        costed("spatial.broadcast_spatial"_op, 1, 0, 0),
        costed("spatial.channel_affine"_op, 2, 0, 0),
        costed("spatial.col2im"_op, 1, 0, 0),
        costed("spatial.concat_channels"_op, 0, 0, 0),
        costed("spatial.im2col"_op, 0, 0, 0),
        costed("spatial.layout_convert"_op, 0, 0, 0),
        costed("spatial.pool_avg"_op, 0, 1, 1),
        costed("spatial.pool_avg_backward"_op, 0, 2, 0),
        costed("spatial.pool_max"_op, 0, 1, 0),
        costed("spatial.pool_max_backward"_op, 0, 0, 1),
        costed("spatial.slice_channels"_op, 0, 0, 0),
        costed("spatial.upsample_nearest"_op, 0, 0, 0),
        costed("spatial.upsample_nearest_backward"_op, 1, 0, 0),
        costed("spatial.window_dot"_op, 0, 0, 0),
        costed("spatial.window_dot_backward"_op, 0, 0, 0),
        costed("spatial.window_dot_weights_backward"_op, 0, 0, 0)
        };

        /// @brief The rows ordered by id, which find_cost_row searches.
        constexpr std::array K_ROWS = sorted_by_id(K_NAMED_ROWS);

        /** @brief Returns a negative extent as 0, so a malformed shape costs nothing it could overflow. */
        constexpr std::uint64_t extent(std::int64_t value) noexcept
        {
            return value < 1 ? 0U : static_cast<std::uint64_t>(value);
        }

        /** @brief Returns a * b, or the largest value when the product does not fit. */
        constexpr std::uint64_t times(std::uint64_t a, std::uint64_t b) noexcept
        {
            if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a)
                return std::numeric_limits<std::uint64_t>::max();
            return a * b;
        }

        /** @brief Returns a + b, or the largest value when the sum does not fit. */
        constexpr std::uint64_t plus(std::uint64_t a, std::uint64_t b) noexcept
        {
            if (b > std::numeric_limits<std::uint64_t>::max() - a)
                return std::numeric_limits<std::uint64_t>::max();
            return a + b;
        }

        /** @brief Returns the smallest power of two that is at least @p value; 1 for 0 and 1. */
        constexpr std::uint64_t padded_size(std::uint64_t value) noexcept
        {
            std::uint64_t size = 1;
            while (size < value && size < (std::uint64_t{1} << 62))
                size <<= 1;
            return size;
        }

        /** @brief Returns log2 of a power of two. */
        constexpr std::uint64_t log2_of(std::uint64_t power) noexcept
        {
            std::uint64_t bits = 0;
            while (power > 1)
            {
                power >>= 1;
                ++bits;
            }
            return bits;
        }

        /** @brief Returns the operations of one complex transform line of @p n points. */
        constexpr std::uint64_t line_flops(std::uint64_t n) noexcept
        {
            if (n <= 1)
                return 0;
            if ((n & (n - 1)) == 0)
                return times(times(5U, n), log2_of(n));
            const std::uint64_t m = padded_size(2 * n - 1);
            return plus(plus(times(times(10U, m), log2_of(m)), times(6U, m)), times(14U, n));
        }

        /** @brief Builds a costed Cost of @p flops for a dtype's ceiling class. */
        constexpr Cost stated(std::uint64_t flops, Core::DataType dtype) noexcept
        {
            Cost cost;
            cost.flops = flops;
            cost.dtype = ceiling_class_name(dtype);
            return cost;
        }

        /** @brief Applies the rule of each shape family to the row of an operation. */
        struct Rules
        {
            /// @brief The row of the operation being costed.
            const CostRow& row;

            /** @brief Costs nothing: a call site that stated no shape has no rule to apply. */
            Cost operator()(std::monostate) const noexcept
            {
                return Cost::uncosted(UncostedReason::NoShape);
            }

            /** @brief Costs 2 per multiply-add of a real product and 8 of a complex one, plus the epilogue. */
            Cost operator()(const GemmShape& shape) const noexcept
            {
                const std::uint64_t outputs =
                    times(times(extent(shape.m), extent(shape.n)), extent(shape.batch));
                const std::uint64_t per_madd = Core::is_complex_dtype(shape.dtype) ? 8U : 2U;
                const std::uint64_t product = times(times(outputs, extent(shape.k)), per_madd);
                return stated(plus(product, times(outputs, shape.epilogue_flops_per_element)),
                              shape.dtype);
            }

            /** @brief Costs 2 per multiply-add over the channels of one group, plus the epilogue per output. */
            Cost operator()(const ConvolutionShape& shape) const noexcept
            {
                const std::uint64_t groups = shape.groups < 1 ? 1U : extent(shape.groups);
                const std::uint64_t outputs = times(
                    times(extent(shape.batch), extent(shape.out_h)),
                    times(extent(shape.out_w), extent(shape.out_channels)));
                const std::uint64_t taps = times(
                    times(extent(shape.in_channels) / groups, extent(shape.kernel_h)),
                    extent(shape.kernel_w));
                const std::uint64_t per_madd = Core::is_complex_dtype(shape.dtype) ? 8U : 2U;
                const std::uint64_t product = times(times(outputs, taps), per_madd);
                return stated(plus(product, times(outputs, shape.epilogue_flops_per_output)),
                              shape.dtype);
            }

            /** @brief Costs the row's per-element weight for each element. */
            Cost operator()(const ElementwiseShape& shape) const noexcept
            {
                return stated(times(extent(shape.count), row.per_element), shape.dtype);
            }

            /** @brief Costs the per-element work, the combines that fold the inputs, and the per-output work. */
            Cost operator()(const ReductionShape& shape) const noexcept
            {
                const std::uint64_t inputs = extent(shape.inputs);
                const std::uint64_t outputs = extent(shape.outputs);
                const std::uint64_t combines = inputs > outputs ? inputs - outputs : 0U;
                const std::uint64_t flops = plus(plus(times(inputs, row.per_element),
                                                      times(combines, row.per_combine)),
                                                 times(outputs, row.per_output));
                return stated(flops, shape.dtype);
            }

            /** @brief Costs one combine per tap of each output and the per-output work. */
            Cost operator()(const WindowShape& shape) const noexcept
            {
                const std::uint64_t outputs = extent(shape.outputs);
                const std::uint64_t flops = plus(times(times(outputs, extent(shape.taps)),
                                                       row.per_combine),
                                                 times(outputs, row.per_output));
                return stated(flops, shape.dtype);
            }

            /**
             * @brief Costs the row pass and the column pass of the lines.
             *
             * A line of n points is `5 n log2(n)` when n is a power of two, else a Bluestein line
             * of padded size m: `10 m log2(m) + 6 m + 14 n`. A real even row is n/2 points plus
             * `5 n`. The forward and the inverse count alike.
             */
            Cost operator()(const TransformShape& shape) const noexcept
            {
                const std::uint64_t batch = extent(shape.batch);
                const std::uint64_t rows = extent(shape.rows);
                const std::uint64_t columns = extent(shape.columns);
                const bool packed = shape.real && columns >= 2 && columns % 2 == 0;
                const std::uint64_t row_line =
                    packed ? plus(line_flops(columns / 2), times(5U, columns))
                           : line_flops(columns);
                const std::uint64_t width = shape.real ? columns / 2 + 1 : columns;
                const std::uint64_t flops = times(
                    batch, plus(times(rows, row_line), times(width, line_flops(rows))));
                return stated(flops, shape.dtype);
            }

            /** @brief Costs no floating-point operation: the task only moves data. */
            Cost operator()(const CopyShape& shape) const noexcept
            {
                return stated(0, shape.dtype);
            }
        };
    } // namespace

    CostRows cost_rows() noexcept
    {
        return CostRows{K_ROWS.data(), K_ROWS.size()};
    }

    const CostRow* find_cost_row(SushiRuntime::OperationID id) noexcept
    {
        const auto at = std::lower_bound(K_ROWS.begin(), K_ROWS.end(), id,
                                         [](const CostRow& row, SushiRuntime::OperationID key)
                                         { return row.id < key; });
        return at != K_ROWS.end() && at->id == id ? &*at : nullptr;
    }

    Cost cost_of(SushiRuntime::OperationID id, const CostShape& shape) noexcept
    {
        const CostRow* row = find_cost_row(id);
        if (row == nullptr)
            return Cost::uncosted(UncostedReason::NoRule);
        if (row->reason != UncostedReason::None)
            return Cost::uncosted(row->reason);
        return std::visit(Rules{*row}, shape);
    }
} // namespace SushiBLAS
