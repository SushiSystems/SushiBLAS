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
#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/engine/task.hpp>
#include <SushiBLAS/engine/spatial.hpp>
#include <SushiBLAS/engine/spatial/window.hpp>
#include "kernels/spatial.hpp"

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    namespace
    {
        /**
         * @brief Builds the span for an upsample from its two rank-4 operands.
         * @return The span, in element counts, with both operands' strides filled in.
         * @throws std::runtime_error when a rank is not 4, a scale is not positive, or the
         *         extents are not the exact scaled multiples.
         * @see src/README.md
         */
        Spatial::WindowSpan upsample_span_for(const Tensor& in, const Tensor& out,
                                              int64_t scale_h, int64_t scale_w,
                                              const char* name)
        {
            SB_THROW_IF(in.rank != 4,
                        "Spatial operation '{}' needs a rank-4 NHWC input; got rank {}.",
                        name, in.rank);
            SB_THROW_IF(out.rank != 4,
                        "Spatial operation '{}' needs a rank-4 NHWC output; got rank {}.",
                        name, out.rank);
            SB_THROW_IF(scale_h <= 0,
                        "Spatial operation '{}' needs a positive vertical scale; got {}.",
                        name, scale_h);
            SB_THROW_IF(scale_w <= 0,
                        "Spatial operation '{}' needs a positive horizontal scale; got {}.",
                        name, scale_w);

            Spatial::WindowSpan span;
            span.batch = in.shape[0];
            span.in_h = in.shape[1];
            span.in_w = in.shape[2];
            span.channels = in.shape[3];
            span.out_h = out.shape[1];
            span.out_w = out.shape[2];

            SB_THROW_IF(out.shape[0] != span.batch,
                        "Spatial operation '{}' has {} input images and {} output images.",
                        name, span.batch, out.shape[0]);
            SB_THROW_IF(out.shape[3] != span.channels,
                        "Spatial operation '{}' has {} input channels and {} output channels.",
                        name, span.channels, out.shape[3]);
            SB_THROW_IF(span.out_h != span.in_h * scale_h,
                        "Spatial operation '{}' scales {} input rows by {}, which is {} output "
                        "rows, but the output has {}.",
                        name, span.in_h, scale_h, span.in_h * scale_h, span.out_h);
            SB_THROW_IF(span.out_w != span.in_w * scale_w,
                        "Spatial operation '{}' scales {} input columns by {}, which is {} output "
                        "columns, but the output has {}.",
                        name, span.in_w, scale_w, span.in_w * scale_w, span.out_w);

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

    sycl::event SpatialOps::upsample_nearest(const Tensor& in, Tensor& out,
                                             int64_t scale_h, int64_t scale_w)
    {
        static constexpr const char* NAME = "spatial.upsample_nearest";

        SB_THROW_IF(in.dtype != out.dtype,
                    "Spatial operation '{}' has a {} input and a {} output.",
                    NAME, Core::to_string(in.dtype), Core::to_string(out.dtype));

        const Spatial::WindowSpan span = upsample_span_for(in, out, scale_h, scale_w, NAME);

        const void* read_in = in.data_or_null();
        void* write_out = out.data_or_null();

        std::vector<SushiRuntime::Core::ResourceRegion> reads = {};
        if (read_in) reads.push_back(in.region());
        std::vector<SushiRuntime::Core::ResourceRegion> writes = {};
        if (write_out) writes.push_back(out.region());

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = NAME;
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "spatial.upsample_nearest"_op;

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

    sycl::event SpatialOps::upsample_nearest_backward(const Tensor& dy, Tensor& dx,
                                                      int64_t scale_h, int64_t scale_w)
    {
        static constexpr const char* NAME = "spatial.upsample_nearest_backward";

        SB_THROW_IF(dy.dtype != dx.dtype,
                    "Spatial operation '{}' has a {} output gradient and a {} input gradient.",
                    NAME, Core::to_string(dy.dtype), Core::to_string(dx.dtype));

        // dx plays the input role and dy the output role, as in the forward; see src/README.md.
        const Spatial::WindowSpan span = upsample_span_for(dx, dy, scale_h, scale_w, NAME);

        const void* read_dy = dy.data_or_null();
        void* write_dx = dx.data_or_null();

        std::vector<SushiRuntime::Core::ResourceRegion> reads = {};
        if (read_dy) reads.push_back(dy.region());
        std::vector<SushiRuntime::Core::ResourceRegion> writes = {};
        if (write_dx) writes.push_back(dx.region());

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = NAME;
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "spatial.upsample_nearest_backward"_op;

        return Internal::execute_typed_task<Dispatch::Domain::REAL>(
            engine_, meta, reads, writes, dy.dtype, "Spatial operation",
            Internal::ReferenceOperands{read_dy, write_dx},
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
