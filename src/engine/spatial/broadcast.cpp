/**************************************************************************/
/* broadcast.cpp                                                          */
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

#include <vector>
#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/engine/task.hpp>
#include <SushiBLAS/engine/spatial.hpp>
#include <SushiBLAS/engine/spatial/window.hpp>
#include "kernels/spatial.hpp"

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    // None of the three operations here has a sliding window, so none of them
    // goes through Spatial::window_span_for: that helper validates every window
    // origin against a ConvGeometry, and the geometry these operations would
    // have to invent to satisfy it (a 1x1 window at stride 1) is six fields a
    // reader would have to rule out before trusting the call. The spans below
    // are built by hand from the operands' shape[] and strides[], with
    // out_h/out_w set equal to in_h/in_w because these are copies rather than
    // reductions over a window.

    sycl::event SpatialOps::channel_affine(const Tensor& in, const Tensor& gamma,
                                           const Tensor& beta, Tensor& out)
    {
        static constexpr const char* NAME = "spatial.channel_affine";

        SB_THROW_IF(in.rank != 4, "Spatial operation '{}' needs a rank-4 NHWC input; got rank {}.",
                    NAME, in.rank);
        SB_THROW_IF(out.rank != 4, "Spatial operation '{}' needs a rank-4 NHWC output; got rank {}.",
                    NAME, out.rank);
        SB_THROW_IF(gamma.rank != 1, "Spatial operation '{}' needs a rank-1 scale; got rank {}.",
                    NAME, gamma.rank);
        SB_THROW_IF(beta.rank != 1, "Spatial operation '{}' needs a rank-1 shift; got rank {}.",
                    NAME, beta.rank);

        SB_THROW_IF(in.dtype != out.dtype,
                    "Spatial operation '{}' has a {} input and a {} output.",
                    NAME, Core::to_string(in.dtype), Core::to_string(out.dtype));
        SB_THROW_IF(gamma.dtype != in.dtype,
                    "Spatial operation '{}' has a {} input and a {} scale.",
                    NAME, Core::to_string(in.dtype), Core::to_string(gamma.dtype));
        SB_THROW_IF(beta.dtype != in.dtype,
                    "Spatial operation '{}' has a {} input and a {} shift.",
                    NAME, Core::to_string(in.dtype), Core::to_string(beta.dtype));

        for (int32_t d = 0; d < 4; ++d)
        {
            SB_THROW_IF(in.shape[d] != out.shape[d],
                        "Spatial operation '{}' has extent {} on axis {} of the input "
                        "and {} on the output's.",
                        NAME, in.shape[d], d, out.shape[d]);
        }

        const int64_t channels = in.shape[3];
        SB_THROW_IF(gamma.shape[0] != channels,
                    "Spatial operation '{}' has {} channels but a scale of length {}.",
                    NAME, channels, gamma.shape[0]);
        SB_THROW_IF(beta.shape[0] != channels,
                    "Spatial operation '{}' has {} channels but a shift of length {}.",
                    NAME, channels, beta.shape[0]);

        Spatial::WindowSpan span;
        span.batch = in.shape[0];
        span.in_h = in.shape[1];
        span.in_w = in.shape[2];
        span.channels = channels;
        span.out_h = span.in_h;
        span.out_w = span.in_w;
        span.in_batch_stride = in.strides[0];
        span.in_row_stride = in.strides[1];
        span.in_col_stride = in.strides[2];
        span.in_channel_stride = in.strides[3];
        span.out_batch_stride = out.strides[0];
        span.out_row_stride = out.strides[1];
        span.out_col_stride = out.strides[2];
        span.out_channel_stride = out.strides[3];

        // Through their own strides, not assumed contiguous: a parameter vector
        // is routinely a view into a packed parameter arena.
        const int64_t gamma_stride = gamma.strides[0];
        const int64_t beta_stride = beta.strides[0];

        void* read_in = in.data_or_null();
        void* read_gamma = gamma.data_or_null();
        void* read_beta = beta.data_or_null();
        void* write_out = out.data_or_null();

        std::vector<SushiRuntime::Core::ResourceRegion> reads = {};
        if (read_in) reads.push_back(in.region());
        if (read_gamma) reads.push_back(gamma.region());
        if (read_beta) reads.push_back(beta.region());
        std::vector<SushiRuntime::Core::ResourceRegion> writes = {};
        if (write_out) writes.push_back(out.region());

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = NAME;
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "spatial.channel_affine"_op;

        const auto mode = engine_.get_mode();

        return Internal::execute_typed_task<Dispatch::Domain::REAL>(
            engine_, meta, reads, writes, in.dtype, "Spatial operation",
            [=](auto scalar_type, sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);

                auto source = static_cast<const T*>(read_in);
                auto scale = static_cast<const T*>(read_gamma);
                auto shift = static_cast<const T*>(read_beta);
                auto destination = static_cast<T*>(write_out);

                return Internal::dispatch_with_mode(mode, q, deps,
                    [=] { Kernels::Reference::channel_affine<T>(span, source, scale, shift,
                                                                gamma_stride, beta_stride,
                                                                destination); },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                    {
                        return Kernels::SYCL::channel_affine<T>(q2, span, source, scale, shift,
                                                                gamma_stride, beta_stride,
                                                                destination, deps2);
                    });
            });
    }

    sycl::event SpatialOps::broadcast_spatial(const Tensor& in, Tensor& out, double scale)
    {
        static constexpr const char* NAME = "spatial.broadcast_spatial";

        SB_THROW_IF(in.rank != 2, "Spatial operation '{}' needs a rank-2 [N, C] input; got rank {}.",
                    NAME, in.rank);
        SB_THROW_IF(out.rank != 4, "Spatial operation '{}' needs a rank-4 NHWC output; got rank {}.",
                    NAME, out.rank);

        SB_THROW_IF(in.dtype != out.dtype,
                    "Spatial operation '{}' has a {} input and a {} output.",
                    NAME, Core::to_string(in.dtype), Core::to_string(out.dtype));

        SB_THROW_IF(in.shape[0] != out.shape[0],
                    "Spatial operation '{}' has {} input rows and {} output images.",
                    NAME, in.shape[0], out.shape[0]);
        SB_THROW_IF(in.shape[1] != out.shape[3],
                    "Spatial operation '{}' has {} input columns and {} output channels.",
                    NAME, in.shape[1], out.shape[3]);

        Spatial::WindowSpan span;
        span.batch = out.shape[0];
        span.out_h = out.shape[1];
        span.out_w = out.shape[2];
        span.channels = out.shape[3];
        // The input is rank-2, so it has no row or column stride to record; the
        // in_* extents are filled in only so that the span describes one shape
        // rather than two, and the input's two real strides travel to the
        // kernels as arguments.
        span.in_h = span.out_h;
        span.in_w = span.out_w;
        span.out_batch_stride = out.strides[0];
        span.out_row_stride = out.strides[1];
        span.out_col_stride = out.strides[2];
        span.out_channel_stride = out.strides[3];

        const int64_t in_batch_stride = in.strides[0];
        const int64_t in_channel_stride = in.strides[1];

        void* read_in = in.data_or_null();
        void* write_out = out.data_or_null();

        std::vector<SushiRuntime::Core::ResourceRegion> reads = {};
        if (read_in) reads.push_back(in.region());
        std::vector<SushiRuntime::Core::ResourceRegion> writes = {};
        if (write_out) writes.push_back(out.region());

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = NAME;
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "spatial.broadcast_spatial"_op;

        const auto mode = engine_.get_mode();

        return Internal::execute_typed_task<Dispatch::Domain::REAL>(
            engine_, meta, reads, writes, in.dtype, "Spatial operation",
            [=](auto scalar_type, sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);

                auto source = static_cast<const T*>(read_in);
                auto destination = static_cast<T*>(write_out);

                // Narrowed here rather than at the seam: the factor has to be a
                // T for both kernels, and T is not known until this closure is
                // instantiated.
                const T factor = static_cast<T>(scale);

                return Internal::dispatch_with_mode(mode, q, deps,
                    [=] { Kernels::Reference::broadcast_spatial<T>(span, source, in_batch_stride,
                                                                   in_channel_stride, destination,
                                                                   factor); },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                    {
                        return Kernels::SYCL::broadcast_spatial<T>(q2, span, source,
                                                                   in_batch_stride,
                                                                   in_channel_stride, destination,
                                                                   factor, deps2);
                    });
            });
    }

    sycl::event SpatialOps::layout_convert(const Tensor& in, Tensor& out)
    {
        static constexpr const char* NAME = "spatial.layout_convert";

        SB_THROW_IF(in.rank != 4, "Spatial operation '{}' needs a rank-4 input; got rank {}.",
                    NAME, in.rank);
        SB_THROW_IF(out.rank != 4, "Spatial operation '{}' needs a rank-4 output; got rank {}.",
                    NAME, out.rank);

        SB_THROW_IF(in.dtype != out.dtype,
                    "Spatial operation '{}' has a {} input and a {} output.",
                    NAME, Core::to_string(in.dtype), Core::to_string(out.dtype));

        // The logical shape is the same on both sides - what differs is which
        // stride is largest. A caller expresses the transpose by handing in a
        // differently-strided output tensor, never by a direction flag, so an
        // output whose *shape* is permuted is a caller error rather than the
        // way NCHW is requested.
        for (int32_t d = 0; d < 4; ++d)
        {
            SB_THROW_IF(in.shape[d] != out.shape[d],
                        "Spatial operation '{}' has extent {} on axis {} of the input and {} on "
                        "the output's; a layout conversion permutes strides, not shapes.",
                        NAME, in.shape[d], d, out.shape[d]);
        }

        Spatial::WindowSpan span;
        span.batch = in.shape[0];
        span.in_h = in.shape[1];
        span.in_w = in.shape[2];
        span.channels = in.shape[3];
        span.out_h = span.in_h;
        span.out_w = span.in_w;
        span.in_batch_stride = in.strides[0];
        span.in_row_stride = in.strides[1];
        span.in_col_stride = in.strides[2];
        span.in_channel_stride = in.strides[3];
        span.out_batch_stride = out.strides[0];
        span.out_row_stride = out.strides[1];
        span.out_col_stride = out.strides[2];
        span.out_channel_stride = out.strides[3];

        void* read_in = in.data_or_null();
        void* write_out = out.data_or_null();

        std::vector<SushiRuntime::Core::ResourceRegion> reads = {};
        if (read_in) reads.push_back(in.region());
        std::vector<SushiRuntime::Core::ResourceRegion> writes = {};
        if (write_out) writes.push_back(out.region());

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = NAME;
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "spatial.layout_convert"_op;

        const auto mode = engine_.get_mode();

        return Internal::execute_typed_task<Dispatch::Domain::REAL>(
            engine_, meta, reads, writes, in.dtype, "Spatial operation",
            [=](auto scalar_type, sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);

                auto source = static_cast<const T*>(read_in);
                auto destination = static_cast<T*>(write_out);

                return Internal::dispatch_with_mode(mode, q, deps,
                    [=] { Kernels::Reference::layout_convert<T>(span, source, destination); },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                    {
                        return Kernels::SYCL::layout_convert<T>(q2, span, source,
                                                                destination, deps2);
                    });
            });
    }
} // namespace SushiBLAS
