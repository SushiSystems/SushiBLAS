/**************************************************************************/
/* col2im.cpp                                                             */
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
 * @file col2im.cpp
 * @brief Defines SpatialOps::col2im, which records the scatter of a column matrix back onto an
 * image.
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

    TaskHandle SpatialOps::col2im(const Tensor& cols, Tensor& image, const Spatial::ConvGeometry& g,
                                  int64_t out_h, int64_t out_w)
    {
        static constexpr const char* NAME = "spatial.col2im";

        Dispatch::require_dtype(image, cols.dtype, NAME, "image");

        Dispatch::require_device_dtype(cols, NAME, "cols");
        Dispatch::require_device_dtype(image, NAME, "image");

        // image plays the input role in the span; see modules/operations/README.md.
        const Spatial::WindowSpan span = Spatial::window_span_for(image, g, out_h, out_w, NAME);
        const Spatial::ColumnSpan column = Spatial::column_span_for(cols, span, NAME);

        const void* read_cols = cols.data_or_null();
        void* write_image = image.data_or_null();

        OperandSet operands;
        operands.reads(cols).writes(image);

        TaskDescription description;
        description.name = NAME;
        description.operation = "spatial.col2im"_op;

        return Internal::execute_typed_task<Dispatch::Domain::Real>(
            recorder_, description, operands, image.dtype, "Spatial operation",
            [=](auto scalar_type, const KernelPairDispatcher& dispatcher,
                sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);

                auto source = static_cast<const T*>(read_cols);
                auto destination = static_cast<T*>(write_image);

                return dispatcher.run(q, deps,
                    [=] { Kernels::Reference::col2im<T>(span, column, source, destination); },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                    {
                        return Kernels::SYCL::col2im<T>(q2, span, column, source, destination, deps2);
                    });
            });
    }
} // namespace SushiBLAS
