/**************************************************************************/
/* im2col.cpp                                                             */
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
 * @file im2col.cpp
 * @brief Defines the two SpatialOps::im2col overloads, which record the gather of image windows
 * into a column matrix.
 * @author Mustafa Garip
 */

#include <vector>
#include <SushiBLAS/engine/task.hpp>
#include <SushiBLAS/engine/spatial.hpp>
#include <SushiBLAS/engine/spatial/window.hpp>
#include "kernels/spatial.hpp"

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    namespace
    {
        constexpr const char* IM2COL_NAME = "spatial.im2col";
    }

    TaskHandle SpatialOps::im2col(const Tensor& in, Tensor& cols, const Spatial::ConvGeometry& g,
                                  int64_t out_h, int64_t out_w)
    {
        SB_THROW_IF(in.dtype != cols.dtype,
                    "Spatial operation '{}' has a {} input and a {} column matrix.",
                    IM2COL_NAME, Core::to_string(in.dtype), Core::to_string(cols.dtype));

        // The column matrix has no image shape, so the extents are passed in; see src/README.md.
        const Spatial::WindowSpan span = Spatial::window_span_for(in, g, out_h, out_w, IM2COL_NAME);
        const Spatial::ColumnSpan column = Spatial::column_span_for(cols, span, IM2COL_NAME);

        const void* read_in = in.data_or_null();
        void* write_cols = cols.data_or_null();

        OperandSet operands;
        operands.reads(in).writes(cols);

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = IM2COL_NAME;
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "spatial.im2col"_op;

        return Internal::execute_typed_task<Dispatch::Domain::REAL>(
            recorder_, meta, operands, in.dtype, "Spatial operation",
            [=](auto scalar_type, const Internal::KernelPairDispatcher& dispatcher,
                sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);

                auto source = static_cast<const T*>(read_in);
                auto destination = static_cast<T*>(write_cols);

                return dispatcher.run(q, deps,
                    [=] { Kernels::Reference::im2col<T>(span, column, source, destination); },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                    {
                        return Kernels::SYCL::im2col<T>(q2, span, column, source, destination, deps2);
                    });
            });
    }

    TaskHandle SpatialOps::im2col(const Tensor& in, Tensor& cols, const Spatial::ConvGeometry& g)
    {
        SB_THROW_IF(in.rank != 4, "Spatial operation '{}' needs a rank-4 NHWC input; got rank {}.",
                    IM2COL_NAME, in.rank);
        SB_THROW_IF(cols.rank != 2,
                    "Spatial operation '{}' needs a rank-2 column matrix; got rank {}.",
                    IM2COL_NAME, cols.rank);

        const int64_t batch = in.shape[0];
        SB_THROW_IF(batch <= 0 || cols.shape[0] % batch != 0,
                    "Spatial operation '{}' cannot split {} column rows across {} images.",
                    IM2COL_NAME, cols.shape[0], batch);
        const int64_t windows_per_image = cols.shape[0] / batch;

        // This overload assumes symmetric padding; see src/README.md.
        const int64_t out_w = Spatial::output_extent(in.shape[2], g.pad_left, g.pad_left,
                                                     g.dilation_w, g.kernel_w, g.stride_w);
        SB_THROW_IF(out_w <= 0 || windows_per_image % out_w != 0,
                    "Spatial operation '{}' derived {} output columns, which does not divide the "
                    "{} windows per image the column matrix asks for.",
                    IM2COL_NAME, out_w, windows_per_image);
        const int64_t out_h = windows_per_image / out_w;

        return im2col(in, cols, g, out_h, out_w);
    }
} // namespace SushiBLAS
