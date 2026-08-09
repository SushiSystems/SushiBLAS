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
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

#pragma once

/**
 * @file spatial.hpp
 * @brief Windowed and strided movement over an N-D grid.
 *
 * The word "convolution" does not appear in this domain and should not be added
 * to it. `im2col` is a gather, `pool_max` is a windowed reduction,
 * `channel_affine` is a broadcast elementwise. Parameter ownership, training
 * mode and differentiation belong to the layer above; what belongs here is
 * moving numbers around a grid.
 *
 * Every operation in this domain has both a pure-C++ reference kernel and a
 * SYCL one, and the reference is the specification. The two are held to each
 * other by this domain's unit tests, which run both forced modes into separate
 * buffers and compare - not by `ExecutionMode::VERIFY`, which runs both but
 * lets the reference overwrite the device's answer rather than checking it.
 *
 * **Every backward here is a gather, never an atomic scatter.** SushiBLAS links
 * `sushiruntime_deterministic_fp` PUBLIC, and a floating-point atomic add has a
 * combination order that varies between runs - so `col2im` written as "each
 * column entry atomically adds into its pixel" would silently break the
 * library's bit-reproducibility guarantee. Each output element gathers its own
 * contributions in a fixed order instead. This is the same argument
 * `docs/ARCHITECTURE.md` gives for rejecting `sycl::reduction` in the folding
 * reductions, and `tests/functional/regression/test_spatial_determinism.cpp` is
 * what keeps it true.
 */

#include <cstdint>
#include <sycl/sycl.hpp>
#include <SushiBLAS/tensor.hpp>
#include <SushiBLAS/engine/spatial/geometry.hpp>

namespace SushiBLAS
{
    class Engine;

    /**
     * @class SpatialOps
     * @brief Windowed gathers, windowed reductions and channel-wise broadcasts.
     */
    class SpatialOps
    {
        public:
            /**
             * @brief Construct SpatialOps with a reference to the engine.
             * @param e The SushiBLAS engine.
             */
            explicit SpatialOps(Engine& e) : engine_(e) {}

            /**
             * @brief Gathers every window of an NHWC image into one row of a matrix.
             *
             * `[N, H, W, C]` to `[N * H_o * W_o, R * S * C]`, with the patch
             * index running `(r, s, c)` - channel innermost. Out-of-bounds taps
             * are zero; padding is a parameter and never a materialised copy.
             *
             * **This is the overload to call when the padding is asymmetric.**
             * @ref Spatial::ConvGeometry carries `pad_top` and `pad_left` only
             * (see @ref Spatial::output_extent for why), so the trailing padding
             * that decides `H_o` and `W_o` is not recoverable from the geometry
             * alone. A caller that has both extents - which SushiAI's shape
             * inference always does - states them here rather than letting the
             * two-operand form assume symmetry.
             *
             * @param in The rank-4 NHWC input.
             * @param cols The rank-2 column matrix to fill.
             * @param g The window geometry.
             * @param out_h The output's row count.
             * @param out_w The output's column count.
             * @return sycl::event representing task completion.
             * @throws std::runtime_error at record time when the ranks, dtypes
             *         or extents disagree.
             */
            sycl::event im2col(const Tensor& in, Tensor& cols, const Spatial::ConvGeometry& g,
                               int64_t out_h, int64_t out_w);

            /**
             * @brief Gathers every window of an NHWC image into one row of a matrix,
             * deriving the output extents from the column matrix.
             *
             * Convenience over the explicit-extent overload, for the symmetric
             * padding every convolution in the ResNet-18 and YOLOv3-tiny bodies
             * uses: `W_o` follows from @ref Spatial::output_extent with
             * `pad_left` on both sides, and `H_o` from dividing the column
             * matrix's row count by it. Asymmetric padding must use the explicit
             * overload, which is why that one is documented first.
             *
             * @param in The rank-4 NHWC input.
             * @param cols The rank-2 column matrix to fill.
             * @param g The window geometry.
             * @return sycl::event representing task completion.
             * @throws std::runtime_error at record time when the ranks, dtypes
             *         or extents disagree, or the derived width does not divide
             *         the column matrix's row count.
             */
            sycl::event im2col(const Tensor& in, Tensor& cols, const Spatial::ConvGeometry& g);

            /**
             * @brief Scatters an im2col matrix back onto an image by summing overlaps.
             *
             * @ref im2col's transpose, and its gradient: entry `(row, patch)` of
             * @p cols is added into the pixel that `im2col` read it from, and
             * entries whose tap fell outside the image are dropped - which is
             * the same crop a materialised zero-padding would have needed.
             *
             * Implemented as a gather over output pixels rather than a scatter
             * over column entries, so overlapping windows need no floating-point
             * atomic add. See this file's header comment for why that is a hard
             * constraint rather than a preference.
             *
             * Every pixel of @p image is written, including pixels no window
             * read; those receive an explicit zero, so the destination needs no
             * prior clear.
             *
             * @param cols The rank-2 column matrix to read.
             * @param image The rank-4 NHWC image to fill.
             * @param g The window geometry.
             * @param out_h The window grid's row count - `im2col`'s `H_o`.
             * @param out_w The window grid's column count - `im2col`'s `W_o`.
             * @return sycl::event representing task completion.
             * @throws std::runtime_error at record time when the ranks, dtypes
             *         or extents disagree.
             */
            sycl::event col2im(const Tensor& cols, Tensor& image, const Spatial::ConvGeometry& g,
                               int64_t out_h, int64_t out_w);

            /**
             * @brief Takes each window's maximum, and records where it came from.
             *
             * The running maximum starts at the lowest finite value of the
             * element type, never at zero - padding is the absence of an
             * element, not a zero-valued candidate, and a zero-initialised
             * maximum silently turns every all-negative border window into 0.
             * Ties keep the first element in scan order, which is PyTorch's
             * convention and what makes the recorded index single-valued.
             *
             * @p indices must be INT64 and shaped like @p out. It holds the
             * **logical** flat position `((n*H + h)*W + w)*C + c` of each
             * window's maximum, which stays comparable across differently-strided
             * views of the same image; @ref pool_max_backward needs it to route
             * the gradient without a scatter.
             *
             * @param in The rank-4 NHWC input.
             * @param out The rank-4 NHWC output.
             * @param indices The INT64 argmax buffer, shaped like @p out.
             * @param g The window geometry.
             * @return sycl::event representing task completion.
             * @throws std::runtime_error at record time when the ranks, dtypes
             *         or extents disagree, or @p indices is not INT64.
             */
            sycl::event pool_max(const Tensor& in, Tensor& out, Tensor& indices,
                                 const Spatial::ConvGeometry& g);

            /**
             * @brief Routes each output gradient back to the element it was taken from.
             *
             * A gather over the windows that could have selected each input
             * pixel, testing the recorded index rather than scattering through
             * it - so overlapping windows need no floating-point atomic add. A
             * pixel selected by several windows accumulates all of their
             * gradients in a fixed order.
             *
             * @p g must be the geometry the forward used; the index buffer alone
             * does not determine the window set.
             *
             * @param dy The rank-4 NHWC output gradient.
             * @param indices The INT64 argmax buffer @ref pool_max wrote.
             * @param dx The rank-4 NHWC input gradient; every pixel is written,
             *        so it needs no prior clear.
             * @param g The window geometry the forward used.
             * @return sycl::event representing task completion.
             * @throws std::runtime_error at record time when the ranks, dtypes
             *         or extents disagree, or @p indices is not INT64.
             */
            sycl::event pool_max_backward(const Tensor& dy, const Tensor& indices, Tensor& dx,
                                          const Spatial::ConvGeometry& g);

            /**
             * @brief Averages each window, dividing by the window's **area**.
             *
             * The divisor is `kernel_h * kernel_w` and counts padded taps that
             * contributed nothing - PyTorch's `count_include_pad=True`, which is
             * its default and therefore what a network ported from it expects.
             * The alternative, dividing by the in-bounds count, gives different
             * numbers at every border pixel of every padded pool.
             *
             * No index buffer: unlike @ref pool_max, this operation's backward
             * needs nothing the geometry does not already say.
             *
             * @param in The rank-4 NHWC input.
             * @param out The rank-4 NHWC output.
             * @param g The window geometry.
             * @return sycl::event representing task completion.
             * @throws std::runtime_error at record time when the ranks, dtypes
             *         or extents disagree.
             */
            sycl::event pool_avg(const Tensor& in, Tensor& out, const Spatial::ConvGeometry& g);

            /**
             * @brief Spreads each output gradient evenly over the window it came from.
             *
             * Each pixel gathers `dy / area` from every window that covered it.
             * Kept separate from @ref pool_max_backward rather than folded behind
             * a mode flag: the two share a loop shape and nothing else, since one
             * selects through a recorded index and this one broadcasts a
             * constant.
             *
             * @param dy The rank-4 NHWC output gradient.
             * @param dx The rank-4 NHWC input gradient; every pixel is written,
             *        so it needs no prior clear.
             * @param g The window geometry the forward used.
             * @return sycl::event representing task completion.
             * @throws std::runtime_error at record time when the ranks, dtypes
             *         or extents disagree.
             */
            sycl::event pool_avg_backward(const Tensor& dy, Tensor& dx,
                                          const Spatial::ConvGeometry& g);

            /**
             * @brief Repeats each input pixel over a `scale_h` by `scale_w` block.
             *
             * `out[n, h_o, w_o, c] = in[n, h_o / scale_h, w_o / scale_w, c]`.
             *
             * Carries no @ref Spatial::ConvGeometry: this operation has two
             * numbers, not eight, and handing it a geometry it does not read
             * would be six fields a reader has to rule out before trusting the
             * call. It builds a @ref Spatial::WindowSpan for the strides alone.
             *
             * @param in The rank-4 NHWC input.
             * @param out The rank-4 NHWC output; its extents must be exactly
             *        `scale_h * H` by `scale_w * W`.
             * @param scale_h The vertical repeat factor; must be positive.
             * @param scale_w The horizontal repeat factor; must be positive.
             * @return sycl::event representing task completion.
             * @throws std::runtime_error at record time when the ranks, dtypes
             *         or extents disagree, or a scale is not positive.
             */
            sycl::event upsample_nearest(const Tensor& in, Tensor& out,
                                         int64_t scale_h, int64_t scale_w);

            /**
             * @brief Sums each output block back onto the pixel that produced it.
             *
             * Each input pixel gathers the fixed `scale_h * scale_w` output
             * elements that read it, in ascending order - so this needs no
             * atomic add and no index buffer.
             *
             * @param dy The rank-4 NHWC output gradient.
             * @param dx The rank-4 NHWC input gradient; every pixel is written.
             * @param scale_h The vertical repeat factor the forward used.
             * @param scale_w The horizontal repeat factor the forward used.
             * @return sycl::event representing task completion.
             * @throws std::runtime_error at record time when the ranks, dtypes
             *         or extents disagree, or a scale is not positive.
             */
            sycl::event upsample_nearest_backward(const Tensor& dy, Tensor& dx,
                                                  int64_t scale_h, int64_t scale_w);

            /**
             * @brief Copies all of @p src's channels into @p dst starting at an offset.
             *
             * The write half of a channel concatenation: call it once per source,
             * each with the offset its channels begin at. `src` and `dst` must
             * agree on `N`, `H` and `W`.
             *
             * **Why not a zero-copy view.** A slice's gradient has to be a
             * materialised operand for the next node, and a graph whose values
             * are sometimes views and sometimes buffers is a distinction the
             * memory planner would have to carry everywhere. The copy is paid
             * once; the complexity would be paid forever.
             *
             * **A known conservatism.** In NHWC a channel range is strided
             * rather than a contiguous byte interval, so two concatenations into
             * one destination produce overlapping `Tensor::region()`s and
             * SushiRuntime's dependency tracker serialises them. That is correct
             * and possibly slower than necessary. Measure before addressing it.
             *
             * @param src The rank-4 NHWC source; all of its channels are copied.
             * @param dst The rank-4 NHWC destination.
             * @param dst_channel_offset The first destination channel to write.
             * @return sycl::event representing task completion.
             * @throws std::runtime_error at record time when the ranks, dtypes
             *         or spatial extents disagree, or the range runs past
             *         @p dst's channel count.
             */
            sycl::event concat_channels(const Tensor& src, Tensor& dst,
                                        int64_t dst_channel_offset);

            /**
             * @brief Copies a channel range out of @p src, filling all of @p dst.
             *
             * @ref concat_channels's inverse, and its backward. One kernel serves
             * both; two names exist because `slice_channels(dy, offset, dx)` says
             * what it means and a shared `copy_channels(dy, offset, dx, 0, count)`
             * does not.
             *
             * @param src The rank-4 NHWC source.
             * @param src_channel_offset The first source channel to read.
             * @param dst The rank-4 NHWC destination; its channel count decides
             *        how many channels are copied.
             * @return sycl::event representing task completion.
             * @throws std::runtime_error at record time when the ranks, dtypes
             *         or spatial extents disagree, or the range runs past
             *         @p src's channel count.
             */
            sycl::event slice_channels(const Tensor& src, int64_t src_channel_offset,
                                       Tensor& dst);

            /**
             * @brief Applies a per-channel scale and shift.
             *
             * `out[n,h,w,c] = in[n,h,w,c] * gamma[c] + beta[c]`. BatchNorm's
             * apply half, and the whole of BatchNorm in evaluation mode.
             *
             * @param in The rank-4 NHWC input.
             * @param gamma The rank-1 `[C]` scale, addressed through its stride.
             * @param beta The rank-1 `[C]` shift, addressed through its stride.
             * @param out The rank-4 NHWC output.
             * @return sycl::event representing task completion.
             * @throws std::runtime_error at record time when the ranks, dtypes
             *         or extents disagree.
             */
            sycl::event channel_affine(const Tensor& in, const Tensor& gamma,
                                       const Tensor& beta, Tensor& out);

            /**
             * @brief Spreads an `[N, C]` value over `[N, H, W, C]`, scaled.
             *
             * `out[n,h,w,c] = in[n,c] * scale`. Global average pooling's
             * backward, with `scale = 1/(H*W)`.
             *
             * A **per-`(n, c)`** broadcast, which is neither of the two the
             * library already has: `GEMM_BIAS` broadcasts a row vector down every
             * row, and @ref channel_affine broadcasts one value per channel
             * across the whole batch.
             *
             * @param in The rank-2 `[N, C]` input.
             * @param out The rank-4 NHWC output.
             * @param scale The factor every element is multiplied by.
             * @return sycl::event representing task completion.
             * @throws std::runtime_error at record time when the ranks, dtypes
             *         or the N/C extents disagree.
             */
            sycl::event broadcast_spatial(const Tensor& in, Tensor& out, double scale);

            /**
             * @brief Copies a rank-4 tensor element by logical position.
             *
             * The boundary transpose between NCHW and NHWC. **Its direction is
             * not a flag**: it reads both tensors' strides and copies by logical
             * `(n, c, h, w)` position, so NCHW-to-NHWC and NHWC-to-NCHW are the
             * same call with the operands' strides swapped.
             *
             * This buys interoperability - NCHW data and NCHW checkpoints in -
             * and is not an NCHW implementation. Two O(N*C*H*W) passes per step
             * are immaterial beside the convolutions they bracket, which is
             * exactly the trade PyTorch's `channels_last` makes.
             *
             * @param in The rank-4 input, in whatever layout its strides describe.
             * @param out The rank-4 output, in whatever layout its strides
             *        describe; its logical shape must match @p in's.
             * @return sycl::event representing task completion.
             * @throws std::runtime_error at record time when the ranks, dtypes
             *         or logical shapes disagree.
             */
            sycl::event layout_convert(const Tensor& in, Tensor& out);

        private:
            Engine& engine_;
    };

} // namespace SushiBLAS
