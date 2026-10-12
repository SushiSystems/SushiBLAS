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
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/spatial/spatial.hpp>
#include <SushiBLAS/spatial/window.hpp>
#include <SushiBLAS/spatial_kernels/spatial.hpp>
#include <SushiBLAS/engine/task_description.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Literals;

    namespace
    {
        constexpr const char* K_IM2COL_NAME = "spatial.im2col";
    }

    TaskHandle SpatialOps::im2col(const Tensor& in, Tensor& cols, const Spatial::ConvGeometry& g,
                                  int64_t out_h, int64_t out_w)
    {
        Dispatch::require_dtype(cols, in.dtype, K_IM2COL_NAME, "cols");

        Dispatch::require_device_dtype(in, K_IM2COL_NAME, "in");
        Dispatch::require_device_dtype(cols, K_IM2COL_NAME, "cols");

        // The column matrix has no image shape, so the extents are passed in; see modules/operations/README.md.
        const Spatial::WindowSpan span = Spatial::window_span_for(in, g, out_h, out_w, K_IM2COL_NAME);
        const Spatial::ColumnSpan column = Spatial::column_span_for(cols, span, K_IM2COL_NAME);

        const void* read_in = in.data_or_null();
        void* write_cols = cols.data_or_null();

        OperandSet operands;
        operands.reads(in).writes(cols);

        TaskDescription description;
        description.name = K_IM2COL_NAME;
        description.operation = "spatial.im2col"_op;

        return Internal::execute_typed_task<Dispatch::Domain::Real>(
            recorder_, description, operands, in.dtype, "Spatial operation",
            [=](auto scalar_type, const KernelPairDispatcher& dispatcher,
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
        Dispatch::require_rank(in, 4, K_IM2COL_NAME, "in");
        Dispatch::require_rank(cols, 2, K_IM2COL_NAME, "cols");

        const int64_t batch = in.shape[0];
        Dispatch::require_operand(batch > 0 && cols.shape[0] % batch == 0, K_IM2COL_NAME, "cols",
                                  "has {} rows, which {} images cannot share evenly",
                                  cols.shape[0], batch);
        const int64_t windows_per_image = cols.shape[0] / batch;

        // This overload assumes symmetric padding; see modules/operations/README.md.
        const int64_t out_w = Spatial::output_extent(in.shape[2], g.pad_left, g.pad_left,
                                                     g.dilation_w, g.kernel_w, g.stride_w);
        Dispatch::require_operand(out_w > 0 && windows_per_image % out_w == 0, K_IM2COL_NAME, "cols",
                                  "asks for {} windows per image, which {} output columns do not "
                                  "divide",
                                  windows_per_image, out_w);
        const int64_t out_h = windows_per_image / out_w;

        return im2col(in, cols, g, out_h, out_w);
    }
} // namespace SushiBLAS
