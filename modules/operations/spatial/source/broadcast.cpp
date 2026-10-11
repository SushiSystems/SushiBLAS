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
#include <SushiBLAS/engine/task.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/spatial/spatial.hpp>
#include <SushiBLAS/spatial/window.hpp>
#include <SushiBLAS/spatial_kernels/spatial.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Literals;

    // The spans here are built by hand, not by Spatial::window_span_for; see modules/operations/README.md.

    TaskHandle SpatialOps::channel_affine(const Tensor& in, const Tensor& gamma,
                                          const Tensor& beta, Tensor& out)
    {
        static constexpr const char* NAME = "spatial.channel_affine";

        Dispatch::require_rank(in, 4, NAME, "in");
        Dispatch::require_rank(out, 4, NAME, "out");
        Dispatch::require_rank(gamma, 1, NAME, "gamma");
        Dispatch::require_rank(beta, 1, NAME, "beta");

        Dispatch::require_dtype(out, in.dtype, NAME, "out");
        Dispatch::require_dtype(gamma, in.dtype, NAME, "gamma");
        Dispatch::require_dtype(beta, in.dtype, NAME, "beta");

        Dispatch::require_device_dtype(in, NAME, "in");
        Dispatch::require_device_dtype(gamma, NAME, "gamma");
        Dispatch::require_device_dtype(beta, NAME, "beta");
        Dispatch::require_device_dtype(out, NAME, "out");

        for (int32_t d = 0; d < 4; ++d)
        {
            Dispatch::require_operand(in.shape[d] == out.shape[d], NAME, "out",
                                      "has extent {} on axis {}, but 'in' has {}",
                                      out.shape[d], d, in.shape[d]);
        }

        const int64_t channels = in.shape[3];
        Dispatch::require_extent(gamma.shape[0], channels, "elements", NAME, "gamma");
        Dispatch::require_extent(beta.shape[0], channels, "elements", NAME, "beta");

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

        // gamma and beta are addressed through their own strides; see modules/operations/README.md.
        const int64_t gamma_stride = gamma.strides[0];
        const int64_t beta_stride = beta.strides[0];

        const void* read_in = in.data_or_null();
        const void* read_gamma = gamma.data_or_null();
        const void* read_beta = beta.data_or_null();
        void* write_out = out.data_or_null();

        OperandSet operands;
        operands.reads(in).reads(gamma).reads(beta).writes(out);

        SushiRuntime::Internal::Graph::TaskMetadata meta;
        meta.name = NAME;
        meta.task_type = SushiRuntime::Internal::Graph::TaskType::MathOp;
        meta.op_id = "spatial.channel_affine"_op;

        return Internal::execute_typed_task<Dispatch::Domain::Real>(
            recorder_, meta, operands, in.dtype, "Spatial operation",
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

    TaskHandle SpatialOps::broadcast_spatial(const Tensor& in, Tensor& out, double scale)
    {
        static constexpr const char* NAME = "spatial.broadcast_spatial";

        Dispatch::require_rank(in, 2, NAME, "in");
        Dispatch::require_rank(out, 4, NAME, "out");

        Dispatch::require_dtype(out, in.dtype, NAME, "out");

        Dispatch::require_device_dtype(in, NAME, "in");
        Dispatch::require_device_dtype(out, NAME, "out");

        Dispatch::require_extent(out.shape[0], in.shape[0], "images", NAME, "out");
        Dispatch::require_extent(out.shape[3], in.shape[1], "channels", NAME, "out");

        Spatial::WindowSpan span;
        span.batch = out.shape[0];
        span.out_h = out.shape[1];
        span.out_w = out.shape[2];
        span.channels = out.shape[3];
        // The rank-2 input has no row or column stride, so in_* mirror out_*; see modules/operations/README.md.
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

        OperandSet operands;
        operands.reads(in).writes(out);

        SushiRuntime::Internal::Graph::TaskMetadata meta;
        meta.name = NAME;
        meta.task_type = SushiRuntime::Internal::Graph::TaskType::MathOp;
        meta.op_id = "spatial.broadcast_spatial"_op;

        return Internal::execute_typed_task<Dispatch::Domain::Real>(
            recorder_, meta, operands, in.dtype, "Spatial operation",
            [=](auto scalar_type, const Internal::KernelPairDispatcher& dispatcher,
                sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);

                auto source = static_cast<const T*>(read_in);
                auto destination = static_cast<T*>(write_out);

                // T is known only in this closure, so scale is narrowed here; see modules/operations/README.md.
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

    TaskHandle SpatialOps::layout_convert(const Tensor& in, Tensor& out)
    {
        static constexpr const char* NAME = "spatial.layout_convert";

        Dispatch::require_rank(in, 4, NAME, "in");
        Dispatch::require_rank(out, 4, NAME, "out");

        Dispatch::require_dtype(out, in.dtype, NAME, "out");

        Dispatch::require_device_dtype(in, NAME, "in");
        Dispatch::require_device_dtype(out, NAME, "out");

        // Both sides share one logical shape and differ only in strides; see modules/operations/README.md.
        for (int32_t d = 0; d < 4; ++d)
        {
            Dispatch::require_operand(in.shape[d] == out.shape[d], NAME, "out",
                                      "has extent {} on axis {}, but 'in' has {}; a layout "
                                      "conversion permutes strides, not shapes",
                                      out.shape[d], d, in.shape[d]);
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

        OperandSet operands;
        operands.reads(in).writes(out);

        SushiRuntime::Internal::Graph::TaskMetadata meta;
        meta.name = NAME;
        meta.task_type = SushiRuntime::Internal::Graph::TaskType::MathOp;
        meta.op_id = "spatial.layout_convert"_op;

        return Internal::execute_typed_task<Dispatch::Domain::Real>(
            recorder_, meta, operands, in.dtype, "Spatial operation",
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
