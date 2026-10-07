/**************************************************************************/
/* spatial.hpp                                                            */
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
 * @file spatial.hpp
 * @brief Declares SpatialOps, the windowed and strided movement operations over an NHWC grid.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/tensor.hpp>
#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/engine/task_handle.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiBLAS/support/spatial_geometry.hpp>

namespace SushiBLAS
{
    /**
     * @class SpatialOps
     * @brief Records windowed gathers, windowed reductions and channel-wise broadcasts.
     */
    class SpatialOps final
    {
        public:
            /** @brief Builds a view that records through recorder, which must outlive the view. */
            explicit SpatialOps(TaskRecorder& recorder) noexcept : recorder_(recorder) {}

            /** @brief Builds a view that records on engine's own graph. */
            explicit SpatialOps(Engine& engine) noexcept : SpatialOps(engine.get_recorder()) {}

            SpatialOps(const SpatialOps&) = default;
            SpatialOps& operator=(const SpatialOps&) = delete;
            SpatialOps(SpatialOps&&) = default;
            SpatialOps& operator=(SpatialOps&&) = delete;
            ~SpatialOps() = default;

            /**
             * @brief Gathers every window of an NHWC image into one row of a matrix.
             * @param out_h The output's row count.
             * @param out_w The output's column count.
             * @throws std::invalid_argument at record time when the ranks, dtypes or extents disagree.
             * @see include/SushiBLAS/engine/README.md
             */
            TaskHandle im2col(const Tensor& in, Tensor& cols, const Spatial::ConvGeometry& g,
                              int64_t out_h, int64_t out_w);

            /**
             * @brief Gathers every window of an NHWC image into matrix rows, deriving the output extents from cols.
             * @pre The padding is symmetric.
             * @throws std::invalid_argument as the other overload, or when the width does not divide the row count.
             * @see include/SushiBLAS/engine/README.md
             */
            TaskHandle im2col(const Tensor& in, Tensor& cols, const Spatial::ConvGeometry& g);

            /**
             * @brief Scatters an im2col matrix back onto an image by summing overlaps.
             * @param out_h The window grid's row count, im2col's `H_o`.
             * @param out_w The window grid's column count, im2col's `W_o`.
             * @throws std::invalid_argument at record time when the ranks, dtypes or extents disagree.
             * @see include/SushiBLAS/engine/README.md
             */
            TaskHandle col2im(const Tensor& cols, Tensor& image, const Spatial::ConvGeometry& g,
                              int64_t out_h, int64_t out_w);

            /**
             * @brief Takes each window's maximum and records where it came from.
             * @param indices Must be INT64 and shaped like out.
             * @throws std::invalid_argument at record time when the ranks, dtypes or extents disagree.
             * @see include/SushiBLAS/engine/README.md
             */
            TaskHandle pool_max(const Tensor& in, Tensor& out, Tensor& indices,
                                const Spatial::ConvGeometry& g);

            /**
             * @brief Routes each output gradient back to the element it was taken from.
             * @param indices The INT64 argmax buffer pool_max wrote.
             * @param g Must be the geometry the forward used.
             * @throws std::invalid_argument at record time when the ranks, dtypes or extents disagree.
             * @see include/SushiBLAS/engine/README.md
             */
            TaskHandle pool_max_backward(const Tensor& dy, const Tensor& indices, Tensor& dx,
                                         const Spatial::ConvGeometry& g);

            /**
             * @brief Averages each window, dividing by the window's area.
             * @throws std::invalid_argument at record time when the ranks, dtypes or extents disagree.
             * @see include/SushiBLAS/engine/README.md
             */
            TaskHandle pool_avg(const Tensor& in, Tensor& out, const Spatial::ConvGeometry& g);

            /**
             * @brief Spreads each output gradient evenly over the window it came from.
             * @param g Must be the geometry the forward used.
             * @throws std::invalid_argument at record time when the ranks, dtypes or extents disagree.
             * @see include/SushiBLAS/engine/README.md
             */
            TaskHandle pool_avg_backward(const Tensor& dy, Tensor& dx,
                                         const Spatial::ConvGeometry& g);

            /**
             * @brief Sums each channel's window weighted by that channel's taps.
             * @param weights The rank-3 `[kernel_h, kernel_w, C]` taps.
             * @param out Its extents state `H_o` and `W_o`; a tap outside the input adds nothing.
             * @throws std::invalid_argument at record time when the ranks, dtypes or extents disagree, or out overlaps an input.
             * @see include/SushiBLAS/engine/README.md
             */
            TaskHandle window_dot(const Tensor& in, const Tensor& weights, Tensor& out,
                                  const Spatial::ConvGeometry& g);

            /**
             * @brief Gathers onto each input element the gradients of the windows that read it, each times its tap.
             * @param weights The taps the forward used.
             * @param g Must be the geometry the forward used.
             * @throws std::invalid_argument at record time when the ranks, dtypes or extents disagree, or dx overlaps an input.
             * @see include/SushiBLAS/engine/README.md
             */
            TaskHandle window_dot_backward(const Tensor& dy, const Tensor& weights, Tensor& dx,
                                           const Spatial::ConvGeometry& g);

            /**
             * @brief Sums, for each tap of each channel, every output gradient times the input element that tap read.
             * @param in The forward's input.
             * @param dweights The rank-3 `[kernel_h, kernel_w, C]` gradient of the taps.
             * @param g Must be the geometry the forward used.
             * @throws std::invalid_argument at record time when the ranks, dtypes or extents disagree, or dweights overlaps an input.
             * @see include/SushiBLAS/engine/README.md
             */
            TaskHandle window_dot_weights_backward(const Tensor& dy, const Tensor& in,
                                                   Tensor& dweights,
                                                   const Spatial::ConvGeometry& g);

            /**
             * @brief Repeats each input pixel over a scale_h by scale_w block.
             * @param out Its extents must be exactly `scale_h * H` by `scale_w * W`.
             * @param scale_h Must be positive, as must scale_w.
             * @throws std::invalid_argument at record time when the ranks, dtypes or extents disagree.
             * @see include/SushiBLAS/engine/README.md
             */
            TaskHandle upsample_nearest(const Tensor& in, Tensor& out,
                                        int64_t scale_h, int64_t scale_w);

            /**
             * @brief Sums each output block back onto the pixel that produced it.
             * @param scale_h The factor the forward used, as is scale_w; both must be positive.
             * @throws std::invalid_argument at record time when the ranks, dtypes or extents disagree.
             * @see include/SushiBLAS/engine/README.md
             */
            TaskHandle upsample_nearest_backward(const Tensor& dy, Tensor& dx,
                                                 int64_t scale_h, int64_t scale_w);

            /**
             * @brief Copies all of src's channels into dst starting at a channel offset.
             * @pre src and dst agree on N, H and W.
             * @throws std::invalid_argument at record time, also when the range runs past dst's channel count.
             * @see include/SushiBLAS/engine/README.md
             */
            TaskHandle concat_channels(const Tensor& src, Tensor& dst,
                                       int64_t dst_channel_offset);

            /**
             * @brief Copies a channel range out of src, filling all of dst.
             * @param dst Its channel count decides how many channels are copied.
             * @throws std::invalid_argument at record time, also when the range runs past src's channel count.
             * @see include/SushiBLAS/engine/README.md
             */
            TaskHandle slice_channels(const Tensor& src, int64_t src_channel_offset,
                                      Tensor& dst);

            /**
             * @brief Applies a per-channel scale and shift: `out[n,h,w,c] = in[n,h,w,c] * gamma[c] + beta[c]`.
             * @param gamma The rank-1 `[C]` scale.
             * @param beta The rank-1 `[C]` shift.
             * @throws std::invalid_argument at record time when the ranks, dtypes or extents disagree.
             * @see include/SushiBLAS/engine/README.md
             */
            TaskHandle channel_affine(const Tensor& in, const Tensor& gamma,
                                      const Tensor& beta, Tensor& out);

            /**
             * @brief Spreads an `[N, C]` value over `[N, H, W, C]`: `out[n,h,w,c] = in[n,c] * scale`.
             * @throws std::invalid_argument at record time when the ranks, dtypes or the N/C extents disagree.
             * @see include/SushiBLAS/engine/README.md
             */
            TaskHandle broadcast_spatial(const Tensor& in, Tensor& out, double scale);

            /**
             * @brief Copies a rank-4 tensor element by logical `(n, c, h, w)` position.
             * @param out Its logical shape must match in's.
             * @throws std::invalid_argument at record time when the ranks, dtypes or logical shapes disagree.
             * @see include/SushiBLAS/engine/README.md
             */
            TaskHandle layout_convert(const Tensor& in, Tensor& out);

        private:
            /// @brief The recorder every operation of this view records through.
            TaskRecorder& recorder_;
    };

} // namespace SushiBLAS
