/****************************************************************/
/* broadcast.hpp                                                */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

#pragma once

/**
 * @file broadcast.hpp
 * @brief The device twins of the three broadcast copies.
 *
 * **Nothing in this file accumulates.** Each work-item writes one output
 * element from one input element and at most one per-channel scalar; there is
 * no reduction, no running total and nothing whose combination order could vary
 * between two runs or differ from the reference kernel's. No `sycl::reduction`,
 * no `sycl::reduce_over_group` and no floating-point `sycl::atomic_ref` appears
 * here, and none may be added - but unlike the pooling backwards, that is not a
 * constraint being worked around: these operations have no overlap to combine
 * in the first place.
 */

#include <vector>
#include <cstdint>
#include <cstddef>
#include <sycl/sycl.hpp>
#include <SushiBLAS/engine/launch_1d.hpp>
#include <SushiBLAS/engine/spatial/geometry.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            /**
             * @brief Applies a per-channel scale and shift.
             *
             * One work-item per output element. @p gamma and @p beta are
             * addressed through their own strides; see the reference kernel of
             * the same name.
             *
             * @tparam T The element type.
             * @param q The queue to submit to.
             * @param span The image layout, in element counts.
             * @param in The input image's first element.
             * @param gamma The scale vector's first element.
             * @param beta The shift vector's first element.
             * @param gamma_stride Elements between adjacent entries of @p gamma.
             * @param beta_stride Elements between adjacent entries of @p beta.
             * @param out The output image's first element.
             * @param deps Events this submission must depend on.
             * @return The event for the submitted kernel.
             */
            template<typename T>
            sycl::event channel_affine(sycl::queue& q, const Spatial::WindowSpan& span,
                                       const T* in, const T* gamma, const T* beta,
                                       int64_t gamma_stride, int64_t beta_stride, T* out,
                                       const std::vector<sycl::event>& deps)
            {
                const int64_t total = span.input_elements();
                if (total <= 0) return sycl::event();

                const int64_t channels = span.channels;
                const int64_t in_h = span.in_h;
                const int64_t in_w = span.in_w;
                const int64_t in_batch_stride = span.in_batch_stride;
                const int64_t in_row_stride = span.in_row_stride;
                const int64_t in_col_stride = span.in_col_stride;
                const int64_t in_channel_stride = span.in_channel_stride;
                const int64_t out_batch_stride = span.out_batch_stride;
                const int64_t out_row_stride = span.out_row_stride;
                const int64_t out_col_stride = span.out_col_stride;
                const int64_t out_channel_stride = span.out_channel_stride;
                const int64_t g_stride = gamma_stride;
                const int64_t b_stride = beta_stride;

                const Device::Grid1D grid =
                    Device::plan_1d(Device::profile_for(q), static_cast<std::size_t>(total));
                const std::size_t stride = grid.stride();
                const std::size_t count = static_cast<std::size_t>(total);

                return q.submit([&](sycl::handler& handler)
                {
                    handler.depends_on(deps);
                    handler.parallel_for(grid.nd_range(), [=](sycl::nd_item<1> it)
                    {
                        for (std::size_t flat = it.get_global_id(0); flat < count; flat += stride)
                        {
                            const int64_t i = static_cast<int64_t>(flat);

                            const int64_t c = i % channels;
                            const int64_t hw = i / channels;
                            const int64_t w = hw % in_w;
                            const int64_t nh = hw / in_w;
                            const int64_t h = nh % in_h;
                            const int64_t n = nh / in_h;

                            const T value = in[n * in_batch_stride + h * in_row_stride
                                             + w * in_col_stride + c * in_channel_stride];

                            out[n * out_batch_stride + h * out_row_stride
                                + w * out_col_stride + c * out_channel_stride] =
                                value * gamma[c * g_stride] + beta[c * b_stride];
                        }
                    });
                });
            }

            /**
             * @brief Spreads an `[N, C]` value over `[N, H, W, C]`, scaled.
             *
             * One work-item per output element, each reading the single
             * `(n, c)` entry its position selects. See the reference kernel for
             * why the rank-2 input's strides are parameters.
             *
             * @tparam T The element type.
             * @param q The queue to submit to.
             * @param span The output image's layout, in element counts.
             * @param in The rank-2 input's first element.
             * @param in_batch_stride Elements between consecutive rows of @p in.
             * @param in_channel_stride Elements between consecutive entries of
             *        one row of @p in.
             * @param out The output image's first element.
             * @param scale The factor every element is multiplied by.
             * @param deps Events this submission must depend on.
             * @return The event for the submitted kernel.
             */
            template<typename T>
            sycl::event broadcast_spatial(sycl::queue& q, const Spatial::WindowSpan& span,
                                          const T* in, int64_t in_batch_stride,
                                          int64_t in_channel_stride, T* out, T scale,
                                          const std::vector<sycl::event>& deps)
            {
                const int64_t total = span.output_elements();
                if (total <= 0) return sycl::event();

                const int64_t channels = span.channels;
                const int64_t out_h = span.out_h;
                const int64_t out_w = span.out_w;
                const int64_t src_batch_stride = in_batch_stride;
                const int64_t src_channel_stride = in_channel_stride;
                const int64_t out_batch_stride = span.out_batch_stride;
                const int64_t out_row_stride = span.out_row_stride;
                const int64_t out_col_stride = span.out_col_stride;
                const int64_t out_channel_stride = span.out_channel_stride;
                const T factor = scale;

                const Device::Grid1D grid =
                    Device::plan_1d(Device::profile_for(q), static_cast<std::size_t>(total));
                const std::size_t stride = grid.stride();
                const std::size_t count = static_cast<std::size_t>(total);

                return q.submit([&](sycl::handler& handler)
                {
                    handler.depends_on(deps);
                    handler.parallel_for(grid.nd_range(), [=](sycl::nd_item<1> it)
                    {
                        for (std::size_t flat = it.get_global_id(0); flat < count; flat += stride)
                        {
                            const int64_t i = static_cast<int64_t>(flat);

                            const int64_t c = i % channels;
                            const int64_t hw = i / channels;
                            const int64_t w = hw % out_w;
                            const int64_t nh = hw / out_w;
                            const int64_t h = nh % out_h;
                            const int64_t n = nh / out_h;

                            const T value = in[n * src_batch_stride + c * src_channel_stride];

                            out[n * out_batch_stride + h * out_row_stride
                                + w * out_col_stride + c * out_channel_stride] = value * factor;
                        }
                    });
                });
            }

            /**
             * @brief Copies a rank-4 tensor element by logical `(n, c, h, w)` position.
             *
             * One work-item per element, indexing each side through its own four
             * strides. The direction is not a flag; see the reference kernel.
             *
             * @tparam T The element type.
             * @param q The queue to submit to.
             * @param span The two layouts, in element counts.
             * @param in The input's first element.
             * @param out The output's first element.
             * @param deps Events this submission must depend on.
             * @return The event for the submitted kernel.
             */
            template<typename T>
            sycl::event layout_convert(sycl::queue& q, const Spatial::WindowSpan& span,
                                       const T* in, T* out, const std::vector<sycl::event>& deps)
            {
                const int64_t total = span.input_elements();
                if (total <= 0) return sycl::event();

                const int64_t channels = span.channels;
                const int64_t in_h = span.in_h;
                const int64_t in_w = span.in_w;
                const int64_t in_batch_stride = span.in_batch_stride;
                const int64_t in_row_stride = span.in_row_stride;
                const int64_t in_col_stride = span.in_col_stride;
                const int64_t in_channel_stride = span.in_channel_stride;
                const int64_t out_batch_stride = span.out_batch_stride;
                const int64_t out_row_stride = span.out_row_stride;
                const int64_t out_col_stride = span.out_col_stride;
                const int64_t out_channel_stride = span.out_channel_stride;

                const Device::Grid1D grid =
                    Device::plan_1d(Device::profile_for(q), static_cast<std::size_t>(total));
                const std::size_t stride = grid.stride();
                const std::size_t count = static_cast<std::size_t>(total);

                return q.submit([&](sycl::handler& handler)
                {
                    handler.depends_on(deps);
                    handler.parallel_for(grid.nd_range(), [=](sycl::nd_item<1> it)
                    {
                        for (std::size_t flat = it.get_global_id(0); flat < count; flat += stride)
                        {
                            const int64_t i = static_cast<int64_t>(flat);

                            const int64_t c = i % channels;
                            const int64_t hw = i / channels;
                            const int64_t w = hw % in_w;
                            const int64_t nh = hw / in_w;
                            const int64_t h = nh % in_h;
                            const int64_t n = nh / in_h;

                            out[n * out_batch_stride + h * out_row_stride
                                + w * out_col_stride + c * out_channel_stride] =
                                in[n * in_batch_stride + h * in_row_stride
                                   + w * in_col_stride + c * in_channel_stride];
                        }
                    });
                });
            }
        }
    }
}
