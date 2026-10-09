/**************************************************************************/
/* channels.cpp                                                           */
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
 * @file channels.cpp
 * @brief Defines SpatialOps::concat_channels and slice_channels, which record the copy of a run of
 * channels.
 * @author Mustafa Garip
 */

#include <vector>
#include <cstdint>
#include <SushiBLAS/engine/task.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/engine/spatial.hpp>
#include "kernels/spatial.hpp"

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    namespace
    {
        /**
         * @brief Checks a channel copy's operands and lays out how to address them.
         * @return The span both kernels address the operands through.
         * @throws std::invalid_argument when the ranks, dtypes or `N`/`H`/`W` extents disagree,
         *         or the range runs past either tensor's channel count.
         * @see source/README.md
         */
        Spatial::WindowSpan channel_copy_span(const char* name,
                                              const Tensor& src, const Tensor& dst,
                                              int64_t src_channel_offset,
                                              int64_t dst_channel_offset, int64_t count)
        {
            Dispatch::require_rank(src, 4, name, "src");
            Dispatch::require_rank(dst, 4, name, "dst");

            Dispatch::require_dtype(dst, src.dtype, name, "dst");

            Dispatch::require_extent(dst.shape[0], src.shape[0], "images", name, "dst");
            Dispatch::require_extent(dst.shape[1], src.shape[1], "rows", name, "dst");
            Dispatch::require_extent(dst.shape[2], src.shape[2], "columns", name, "dst");

            Dispatch::require_operand(count >= 0, name, "count",
                                      "is {}; a channel count cannot be negative", count);

            Dispatch::require_operand(
                src_channel_offset >= 0 && src_channel_offset + count <= src.shape[3], name, "src",
                "has {} channels, but channels [{}, {}) were asked for", src.shape[3],
                src_channel_offset, src_channel_offset + count);

            Dispatch::require_operand(
                dst_channel_offset >= 0 && dst_channel_offset + count <= dst.shape[3], name, "dst",
                "has {} channels, but channels [{}, {}) were asked for", dst.shape[3],
                dst_channel_offset, dst_channel_offset + count);

            Spatial::WindowSpan span;
            span.batch = src.shape[0];
            span.in_h = src.shape[1];
            span.in_w = src.shape[2];
            span.channels = src.shape[3];

            // A copy has no window, so the output grid is the input grid.
            span.out_h = span.in_h;
            span.out_w = span.in_w;

            span.in_batch_stride = src.strides[0];
            span.in_row_stride = src.strides[1];
            span.in_col_stride = src.strides[2];
            span.in_channel_stride = src.strides[3];

            span.out_batch_stride = dst.strides[0];
            span.out_row_stride = dst.strides[1];
            span.out_col_stride = dst.strides[2];
            span.out_channel_stride = dst.strides[3];

            return span;
        }
    } // namespace

    TaskHandle SpatialOps::concat_channels(const Tensor& src, Tensor& dst,
                                           int64_t dst_channel_offset)
    {
        static constexpr const char* NAME = "spatial.concat_channels";

        // All of the source goes in; the caller chooses only where it lands.
        const int64_t count = src.rank == 4 ? src.shape[3] : 0;
        const Spatial::WindowSpan span =
            channel_copy_span(NAME, src, dst, 0, dst_channel_offset, count);

        Dispatch::require_device_dtype(src, NAME, "src");
        Dispatch::require_device_dtype(dst, NAME, "dst");

        const void* read_src = src.data_or_null();
        void* write_dst = dst.data_or_null();

        OperandSet operands;
        operands.reads(src).writes(dst);

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = NAME;
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "spatial.concat_channels"_op;

        // A copy does no arithmetic, so it serves Domain::All; see source/README.md.
        return Internal::execute_typed_task<Dispatch::Domain::All>(
            recorder_, meta, operands, src.dtype, "Spatial operation",
            [=](auto scalar_type, const Internal::KernelPairDispatcher& dispatcher,
                sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);

                auto source = static_cast<const T*>(read_src);
                auto destination = static_cast<T*>(write_dst);

                return dispatcher.run(q, deps,
                    [=]
                    {
                        Kernels::Reference::copy_channel_range<T>(span, 0, dst_channel_offset,
                                                                  count, source, destination);
                    },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                    {
                        return Kernels::SYCL::copy_channel_range<T>(q2, span, 0, dst_channel_offset,
                                                                    count, source, destination,
                                                                    deps2);
                    });
            });
    }

    TaskHandle SpatialOps::slice_channels(const Tensor& src, int64_t src_channel_offset,
                                          Tensor& dst)
    {
        static constexpr const char* NAME = "spatial.slice_channels";

        // The destination's width is the slice's width; nothing else could be.
        const int64_t count = dst.rank == 4 ? dst.shape[3] : 0;
        const Spatial::WindowSpan span =
            channel_copy_span(NAME, src, dst, src_channel_offset, 0, count);

        Dispatch::require_device_dtype(src, NAME, "src");
        Dispatch::require_device_dtype(dst, NAME, "dst");

        const void* read_src = src.data_or_null();
        void* write_dst = dst.data_or_null();

        OperandSet operands;
        operands.reads(src).writes(dst);

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = NAME;
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "spatial.slice_channels"_op;

        return Internal::execute_typed_task<Dispatch::Domain::All>(
            recorder_, meta, operands, src.dtype, "Spatial operation",
            [=](auto scalar_type, const Internal::KernelPairDispatcher& dispatcher,
                sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);

                auto source = static_cast<const T*>(read_src);
                auto destination = static_cast<T*>(write_dst);

                return dispatcher.run(q, deps,
                    [=]
                    {
                        Kernels::Reference::copy_channel_range<T>(span, src_channel_offset, 0,
                                                                  count, source, destination);
                    },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                    {
                        return Kernels::SYCL::copy_channel_range<T>(q2, span, src_channel_offset, 0,
                                                                    count, source, destination,
                                                                    deps2);
                    });
            });
    }
} // namespace SushiBLAS
