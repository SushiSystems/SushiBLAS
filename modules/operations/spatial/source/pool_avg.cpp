/**************************************************************************/
/* pool_avg.cpp                                                           */
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
 * @file pool_avg.cpp
 * @brief Defines SpatialOps::pool_avg and pool_avg_backward, which record average pooling and its
 * gradient.
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

    TaskHandle SpatialOps::pool_avg(const Tensor& in, Tensor& out, const Spatial::ConvGeometry& g)
    {
        static constexpr const char* NAME = "spatial.pool_avg";

        Dispatch::require_dtype(out, in.dtype, NAME, "out");

        Dispatch::require_device_dtype(in, NAME, "in");
        Dispatch::require_device_dtype(out, NAME, "out");

        const Spatial::WindowSpan span = Spatial::pooling_span_for(in, g, out, NAME);

        const void* read_in = in.data_or_null();
        void* write_out = out.data_or_null();

        OperandSet operands;
        operands.reads(in).writes(out);

        TaskDescription description;
        description.name = NAME;
        description.operation = "spatial.pool_avg"_op;
        description.shape = WindowShape{span.output_elements(), g.kernel_h * g.kernel_w, in.dtype};

        return Internal::execute_typed_task<Dispatch::Domain::Real>(
            recorder_, description, operands, in.dtype, "Spatial operation",
            [=](auto scalar_type, const KernelPairDispatcher& dispatcher,
                sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);

                auto source = static_cast<const T*>(read_in);
                auto destination = static_cast<T*>(write_out);

                return dispatcher.run(q, deps,
                    [=] { Kernels::Reference::pool_avg<T>(span, source, destination); },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                    {
                        return Kernels::SYCL::pool_avg<T>(q2, span, source, destination, deps2);
                    });
            });
    }

    TaskHandle SpatialOps::pool_avg_backward(const Tensor& dy, Tensor& dx,
                                             const Spatial::ConvGeometry& g)
    {
        static constexpr const char* NAME = "spatial.pool_avg_backward";

        Dispatch::require_dtype(dx, dy.dtype, NAME, "dx");

        Dispatch::require_device_dtype(dy, NAME, "dy");
        Dispatch::require_device_dtype(dx, NAME, "dx");

        // dx plays the input role and dy the output role, as in the forward; see modules/operations/README.md.
        const Spatial::WindowSpan span = Spatial::pooling_span_for(dx, g, dy, NAME);

        const void* read_dy = dy.data_or_null();
        void* write_dx = dx.data_or_null();

        OperandSet operands;
        operands.reads(dy).writes(dx);

        TaskDescription description;
        description.name = NAME;
        description.operation = "spatial.pool_avg_backward"_op;
        description.shape = WindowShape{span.output_elements(), g.kernel_h * g.kernel_w, dy.dtype};

        return Internal::execute_typed_task<Dispatch::Domain::Real>(
            recorder_, description, operands, dy.dtype, "Spatial operation",
            [=](auto scalar_type, const KernelPairDispatcher& dispatcher,
                sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);

                auto source = static_cast<const T*>(read_dy);
                auto destination = static_cast<T*>(write_dx);

                return dispatcher.run(q, deps,
                    [=] { Kernels::Reference::pool_avg_backward<T>(span, source, destination); },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                    {
                        return Kernels::SYCL::pool_avg_backward<T>(q2, span, source,
                                                                   destination, deps2);
                    });
            });
    }
} // namespace SushiBLAS
