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
 * @file broadcast.cpp
 * @brief Defines SpatialOps::channel_affine, broadcast_spatial and layout_convert, which record
 * the three broadcast copies.
 * @author Mustafa Garip
 */

#include <vector>
#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/engine/task.hpp>
#include <SushiBLAS/engine/spatial.hpp>
#include <SushiBLAS/engine/spatial/window.hpp>
#include "kernels/spatial.hpp"

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    // The spans here are built by hand, not by Spatial::window_span_for; see src/README.md.

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

        // gamma and beta are addressed through their own strides; see src/README.md.
        const int64_t gamma_stride = gamma.strides[0];
        const int64_t beta_stride = beta.strides[0];

        const void* read_in = in.data_or_null();
        const void* read_gamma = gamma.data_or_null();
        const void* read_beta = beta.data_or_null();
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

        return Internal::execute_typed_task<Dispatch::Domain::REAL>(
            engine_, meta, reads, writes, in.dtype, "Spatial operation",
            Internal::ReferenceOperands{read_in, read_gamma, read_beta, write_out},
            [=](auto scalar_type, const Internal::KernelPairDispatcher& dispatcher,
                sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);

                auto source = static_cast<const T*>(read_in);
                auto scale = static_cast<const T*>(read_gamma);
                auto shift = static_cast<const T*>(read_beta);
                auto destination = static_cast<T*>(write_out);

                return dispatcher.run(q, deps,
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
        // The rank-2 input has no row or column stride, so in_* mirror out_*; see src/README.md.
        span.in_h = span.out_h;
        span.in_w = span.out_w;
        span.out_batch_stride = out.strides[0];
        span.out_row_stride = out.strides[1];
        span.out_col_stride = out.strides[2];
        span.out_channel_stride = out.strides[3];

        const int64_t in_batch_stride = in.strides[0];
        const int64_t in_channel_stride = in.strides[1];

        const void* read_in = in.data_or_null();
        void* write_out = out.data_or_null();

        std::vector<SushiRuntime::Core::ResourceRegion> reads = {};
        if (read_in) reads.push_back(in.region());
        std::vector<SushiRuntime::Core::ResourceRegion> writes = {};
        if (write_out) writes.push_back(out.region());

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = NAME;
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "spatial.broadcast_spatial"_op;

        return Internal::execute_typed_task<Dispatch::Domain::REAL>(
            engine_, meta, reads, writes, in.dtype, "Spatial operation",
            Internal::ReferenceOperands{read_in, write_out},
            [=](auto scalar_type, const Internal::KernelPairDispatcher& dispatcher,
                sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);

                auto source = static_cast<const T*>(read_in);
                auto destination = static_cast<T*>(write_out);

                // T is known only in this closure, so scale is narrowed here; see src/README.md.
                const T factor = static_cast<T>(scale);

                return dispatcher.run(q, deps,
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

        // Both sides share one logical shape and differ only in strides; see src/README.md.
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

        const void* read_in = in.data_or_null();
        void* write_out = out.data_or_null();

        std::vector<SushiRuntime::Core::ResourceRegion> reads = {};
        if (read_in) reads.push_back(in.region());
        std::vector<SushiRuntime::Core::ResourceRegion> writes = {};
        if (write_out) writes.push_back(out.region());

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = NAME;
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "spatial.layout_convert"_op;

        return Internal::execute_typed_task<Dispatch::Domain::REAL>(
            engine_, meta, reads, writes, in.dtype, "Spatial operation",
            Internal::ReferenceOperands{read_in, write_out},
            [=](auto scalar_type, const Internal::KernelPairDispatcher& dispatcher,
                sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);

                auto source = static_cast<const T*>(read_in);
                auto destination = static_cast<T*>(write_out);

                return dispatcher.run(q, deps,
                    [=] { Kernels::Reference::layout_convert<T>(span, source, destination); },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                    {
                        return Kernels::SYCL::layout_convert<T>(q2, span, source,
                                                                destination, deps2);
                    });
            });
    }
} // namespace SushiBLAS
