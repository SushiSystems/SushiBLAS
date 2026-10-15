/**************************************************************************/
/* window_dot_weights.hpp                                                 */
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
 * @file window_dot_weights.hpp
 * @brief Defines the SYCL kernel of the weight gradient of the per-channel weighted window sum.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>
#include <sycl/sycl.hpp>
#include <SushiBLAS/arithmetic/accumulator.hpp>
#include <SushiBLAS/reduction/reduce_1d.hpp>
#include <SushiBLAS/reduction/reduce_axis.hpp>
#include <SushiBLAS/geometry/spatial_geometry.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            namespace Detail
            {
                /**
                 * @brief Reads the terms of one tap's gradient: an output gradient times the input element the tap read.
                 * @tparam W The type a term is formed in: Accumulator of the element type.
                 * @see kernels/README.md
                 */
                template<typename T, typename W>
                class TapGradientCursor final
                {
                    public:
                        /**
                         * @brief Binds the cursor to one tap of one channel.
                         * @param row_origin The input row the tap reads for output row 0: `r * dilation_h - pad_top`.
                         * @param col_origin The input column the tap reads for output column 0.
                         */
                        TapGradientCursor(const Spatial::WindowSpan& span, const T* dy, const T* in,
                                          int64_t row_origin, int64_t col_origin, int64_t channel)
                            : span_(span), dy_(dy), in_(in), row_origin_(row_origin),
                              col_origin_(col_origin), channel_(channel)
                        {

                        }

                        /** @brief Returns term k, where k is the flat `(n, h_o, w_o)` window index; zero for a tap outside the input. */
                        [[nodiscard]] W operator()(std::size_t k) const
                        {
                            const int64_t i = static_cast<int64_t>(k);
                            const int64_t w_o = i % span_.out_w;
                            const int64_t nh = i / span_.out_w;
                            const int64_t h_o = nh % span_.out_h;
                            const int64_t n = nh / span_.out_h;

                            const int64_t h_in = h_o * span_.geometry.stride_h + row_origin_;
                            const int64_t w_in = w_o * span_.geometry.stride_w + col_origin_;
                            if (h_in < 0 || h_in >= span_.in_h || w_in < 0 || w_in >= span_.in_w)
                                return W(0);

                            return W(dy_[n * span_.out_batch_stride + h_o * span_.out_row_stride
                                         + w_o * span_.out_col_stride
                                         + channel_ * span_.out_channel_stride])
                                 * W(in_[n * span_.in_batch_stride + h_in * span_.in_row_stride
                                         + w_in * span_.in_col_stride
                                         + channel_ * span_.in_channel_stride]);
                        }

                    private:
                        /// @brief The forward's layout.
                        Spatial::WindowSpan span_;

                        /// @brief The output gradient's first element.
                        const T* dy_;

                        /// @brief The forward input's first element.
                        const T* in_;

                        /// @brief The input row the tap reads for output row 0.
                        int64_t row_origin_;

                        /// @brief The input column the tap reads for output column 0.
                        int64_t col_origin_;

                        /// @brief The channel the tap belongs to.
                        int64_t channel_;
                };

                /**
                 * @brief Opens the taps of a `[kernel_h, kernel_w, channels]` gradient for Reduce::reduce_segments.
                 * @see kernels/README.md
                 */
                template<typename T, typename W>
                class TapGradientOpen final
                {
                    public:
                        /** @brief Binds the opener to the forward's layout and its two read operands. */
                        TapGradientOpen(const Spatial::WindowSpan& span, const T* dy, const T* in)
                            : span_(span), dy_(dy), in_(in)
                        {

                        }

                        /** @brief Returns the cursor of segment `(r * kernel_w + s) * channels + c`. */
                        [[nodiscard]] TapGradientCursor<T, W> operator()(std::size_t segment) const
                        {
                            const int64_t flat = static_cast<int64_t>(segment);
                            const int64_t c = flat % span_.channels;
                            const int64_t rs = flat / span_.channels;
                            const int64_t s = rs % span_.geometry.kernel_w;
                            const int64_t r = rs / span_.geometry.kernel_w;

                            return TapGradientCursor<T, W>(
                                span_, dy_, in_,
                                r * span_.geometry.dilation_h - span_.geometry.pad_top,
                                s * span_.geometry.dilation_w - span_.geometry.pad_left, c);
                        }

                    private:
                        /// @brief The forward's layout.
                        Spatial::WindowSpan span_;

                        /// @brief The output gradient's first element.
                        const T* dy_;

                        /// @brief The forward input's first element.
                        const T* in_;
                };

                /**
                 * @brief Stores one tap's gradient into a `[kernel_h, kernel_w, channels]` tensor.
                 * @see kernels/README.md
                 */
                template<typename T, typename W>
                class TapGradientStore final
                {
                    public:
                        /** @brief Binds the store to the gradient tensor and its addressing. */
                        TapGradientStore(const Spatial::WindowSpan& span, const Spatial::TapSpan& taps,
                                         T* dweights)
                            : channels_(span.channels), kernel_w_(span.geometry.kernel_w),
                              taps_(taps), dweights_(dweights)
                        {

                        }

                        /** @brief Writes value, narrowed to T, to the element of segment `(r * kernel_w + s) * channels + c`. */
                        void operator()(std::size_t segment, W value) const
                        {
                            const int64_t flat = static_cast<int64_t>(segment);
                            const int64_t c = flat % channels_;
                            const int64_t rs = flat / channels_;
                            const int64_t s = rs % kernel_w_;
                            const int64_t r = rs / kernel_w_;

                            dweights_[r * taps_.row_stride + s * taps_.col_stride
                                      + c * taps_.channel_stride] = static_cast<T>(value);
                        }

                    private:
                        /// @brief The channel count.
                        int64_t channels_;

                        /// @brief The window's width in taps.
                        int64_t kernel_w_;

                        /// @brief How the gradient tensor is addressed.
                        Spatial::TapSpan taps_;

                        /// @brief The gradient tensor's first element.
                        T* dweights_;
                };
            } // namespace Detail

            /**
             * @brief Sums, for each tap of each channel, every output gradient times the input element that tap read.
             * @param span The forward's layout; its `in_*` fields describe in and its `out_*` fields dy.
             * @param geometry The launch shape, normally Reduce::plan_axis over `span.patch_elements()` folds of `span.windows()` terms.
             * @param dweights The `[kernel_h, kernel_w, channels]` gradient; every element is written.
             * @see kernels/README.md
             */
            template<typename T>
            [[nodiscard]] sycl::event window_dot_weights_backward(
                sycl::queue& q, const Spatial::WindowSpan& span, const Spatial::TapSpan& taps,
                const Reduce::AxisPlan& geometry, const T* dy, const T* in, T* dweights,
                const std::vector<sycl::event>& deps)
            {
                using W = Accumulator<T>;

                const int64_t segments = span.patch_elements();
                if (segments <= 0)
                    return sycl::event();

                return Reduce::reduce_segments<Reduce::CompensatedSum, W>(
                    q, static_cast<std::size_t>(segments),
                    static_cast<std::size_t>(span.windows()), geometry,
                    Detail::TapGradientOpen<T, W>(span, dy, in),
                    Detail::TapGradientStore<T, W>(span, taps, dweights), deps);
            }
        }
    }
}
