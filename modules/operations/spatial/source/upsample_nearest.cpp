/**************************************************************************/
/* upsample_nearest.cpp                                                   */
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
 * @file upsample_nearest.cpp
 * @brief Defines SpatialOps::upsample_nearest and upsample_nearest_backward, which record nearest-
 * neighbour upsampling and its gradient.
 * @author Mustafa Garip
 */

#include <vector>
#include <cstdint>
#include <SushiBLAS/engine/task.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/spatial/spatial.hpp>
#include <SushiBLAS/spatial/window.hpp>
#include <SushiBLAS/spatial_kernels/spatial.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Literals;

    namespace
    {
        /**
         * @brief Builds the span for an upsample from its two rank-4 operands.
         * @return The span, in element counts, with both operands' strides filled in.
         * @throws std::invalid_argument when a rank is not 4, a scale is not positive, or the
         *         extents are not the exact scaled multiples.
         * @see modules/operations/README.md
         */
        Spatial::WindowSpan upsample_span_for(const Tensor& in, const Tensor& out,
                                              int64_t scale_h, int64_t scale_w,
                                              const char* name)
        {
            Dispatch::require_rank(in, 4, name, "in");
            Dispatch::require_rank(out, 4, name, "out");
            Dispatch::require_operand(scale_h > 0, name, "scale_h",
                                      "is {}; a scale must be positive", scale_h);
            Dispatch::require_operand(scale_w > 0, name, "scale_w",
                                      "is {}; a scale must be positive", scale_w);

            Spatial::WindowSpan span;
            span.batch = in.shape[0];
            span.in_h = in.shape[1];
            span.in_w = in.shape[2];
            span.channels = in.shape[3];
            span.out_h = out.shape[1];
            span.out_w = out.shape[2];

            Dispatch::require_extent(out.shape[0], span.batch, "images", name, "out");
            Dispatch::require_extent(out.shape[3], span.channels, "channels", name, "out");
            Dispatch::require_extent(span.out_h, span.in_h * scale_h, "rows", name, "out");
            Dispatch::require_extent(span.out_w, span.in_w * scale_w, "columns", name, "out");

            span.in_batch_stride = in.strides[0];
            span.in_row_stride = in.strides[1];
            span.in_col_stride = in.strides[2];
            span.in_channel_stride = in.strides[3];

            span.out_batch_stride = out.strides[0];
            span.out_row_stride = out.strides[1];
            span.out_col_stride = out.strides[2];
            span.out_channel_stride = out.strides[3];

            return span;
        }
    } // namespace

    TaskHandle SpatialOps::upsample_nearest(const Tensor& in, Tensor& out,
                                            int64_t scale_h, int64_t scale_w)
    {
        static constexpr const char* NAME = "spatial.upsample_nearest";

        Dispatch::require_dtype(out, in.dtype, NAME, "out");

        Dispatch::require_device_dtype(in, NAME, "in");
        Dispatch::require_device_dtype(out, NAME, "out");

        const Spatial::WindowSpan span = upsample_span_for(in, out, scale_h, scale_w, NAME);

        const void* read_in = in.data_or_null();
        void* write_out = out.data_or_null();

        OperandSet operands;
        operands.reads(in).writes(out);

        SushiRuntime::Internal::Graph::TaskMetadata meta;
        meta.name = NAME;
        meta.task_type = SushiRuntime::Internal::Graph::TaskType::MathOp;
        meta.op_id = "spatial.upsample_nearest"_op;

        return Internal::execute_typed_task<Dispatch::Domain::Real>(
            recorder_, meta, operands, in.dtype, "Spatial operation",
            [=](auto scalar_type, const Internal::KernelPairDispatcher& dispatcher,
                sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);

                auto source = static_cast<const T*>(read_in);
                auto destination = static_cast<T*>(write_out);

                return dispatcher.run(q, deps,
                    [=]
                    {
                        Kernels::Reference::upsample_nearest<T>(span, scale_h, scale_w,
                                                                source, destination);
                    },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                    {
                        return Kernels::SYCL::upsample_nearest<T>(q2, span, scale_h, scale_w,
                                                                  source, destination, deps2);
                    });
            });
    }

    TaskHandle SpatialOps::upsample_nearest_backward(const Tensor& dy, Tensor& dx,
                                                     int64_t scale_h, int64_t scale_w)
    {
        static constexpr const char* NAME = "spatial.upsample_nearest_backward";

        Dispatch::require_dtype(dx, dy.dtype, NAME, "dx");

        Dispatch::require_device_dtype(dy, NAME, "dy");
        Dispatch::require_device_dtype(dx, NAME, "dx");

        // dx plays the input role and dy the output role, as in the forward; see modules/operations/README.md.
        const Spatial::WindowSpan span = upsample_span_for(dx, dy, scale_h, scale_w, NAME);

        const void* read_dy = dy.data_or_null();
        void* write_dx = dx.data_or_null();

        OperandSet operands;
        operands.reads(dy).writes(dx);

        SushiRuntime::Internal::Graph::TaskMetadata meta;
        meta.name = NAME;
        meta.task_type = SushiRuntime::Internal::Graph::TaskType::MathOp;
        meta.op_id = "spatial.upsample_nearest_backward"_op;

        return Internal::execute_typed_task<Dispatch::Domain::Real>(
            recorder_, meta, operands, dy.dtype, "Spatial operation",
            [=](auto scalar_type, const Internal::KernelPairDispatcher& dispatcher,
                sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);

                auto source = static_cast<const T*>(read_dy);
                auto destination = static_cast<T*>(write_dx);

                return dispatcher.run(q, deps,
                    [=]
                    {
                        Kernels::Reference::upsample_nearest_backward<T>(span, scale_h, scale_w,
                                                                         source, destination);
                    },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                    {
                        return Kernels::SYCL::upsample_nearest_backward<T>(q2, span,
                                                                           scale_h, scale_w,
                                                                           source, destination,
                                                                           deps2);
                    });
            });
    }
} // namespace SushiBLAS
