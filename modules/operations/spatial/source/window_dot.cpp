/**************************************************************************/
/* window_dot.cpp                                                         */
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
 * @file window_dot.cpp
 * @brief Defines SpatialOps::window_dot and window_dot_backward, which record the per-channel
 * weighted window sum and its input gradient.
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

    TaskHandle SpatialOps::window_dot(const Tensor& in, const Tensor& weights, Tensor& out,
                                      const Spatial::ConvGeometry& g)
    {
        static constexpr const char* NAME = "spatial.window_dot";

        Dispatch::require_dtype(weights, in.dtype, NAME, "weights");
        Dispatch::require_dtype(out, in.dtype, NAME, "out");

        const Spatial::WindowSpan span = Spatial::window_span_for(in, g, out, NAME);
        const Spatial::TapSpan taps = Spatial::tap_span_for(weights, span, NAME, "weights");

        Dispatch::require_storage(in, NAME, "in");
        Dispatch::require_storage(weights, NAME, "weights");
        Dispatch::require_storage(out, NAME, "out");
        Dispatch::require_device_dtype(in, NAME, "in");
        Dispatch::require_device_dtype(weights, NAME, "weights");
        Dispatch::require_device_dtype(out, NAME, "out");

        // A gather reads its inputs while it writes, so the output may share no byte with them.
        Dispatch::require_operand(!out.region().overlaps(in.region()), NAME, "out",
                                  "overlaps 'in' in memory");
        Dispatch::require_operand(!out.region().overlaps(weights.region()), NAME, "out",
                                  "overlaps 'weights' in memory");

        const void* read_in = in.data_or_null();
        const void* read_weights = weights.data_or_null();
        void* write_out = out.data_or_null();

        OperandSet operands;
        operands.reads(in).reads(weights).writes(out);

        TaskDescription description;
        description.name = NAME;
        description.operation = "spatial.window_dot"_op;

        return Internal::execute_typed_task<Dispatch::Domain::Real>(
            recorder_, description, operands, in.dtype, "Spatial operation",
            [=](auto scalar_type, const Internal::KernelPairDispatcher& dispatcher,
                sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);

                auto source = static_cast<const T*>(read_in);
                auto tap_weights = static_cast<const T*>(read_weights);
                auto destination = static_cast<T*>(write_out);

                return dispatcher.run(q, deps,
                    [=]
                    {
                        Kernels::Reference::window_dot<T>(span, taps, source, tap_weights,
                                                          destination);
                    },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                    {
                        return Kernels::SYCL::window_dot<T>(q2, span, taps, source, tap_weights,
                                                            destination, deps2);
                    });
            });
    }

    TaskHandle SpatialOps::window_dot_backward(const Tensor& dy, const Tensor& weights, Tensor& dx,
                                               const Spatial::ConvGeometry& g)
    {
        static constexpr const char* NAME = "spatial.window_dot_backward";

        Dispatch::require_dtype(weights, dy.dtype, NAME, "weights");
        Dispatch::require_dtype(dx, dy.dtype, NAME, "dx");

        // dx plays the input role and dy the output role, as in the forward; see modules/operations/README.md.
        const Spatial::WindowSpan span = Spatial::window_span_for(dx, g, dy, NAME);
        const Spatial::TapSpan taps = Spatial::tap_span_for(weights, span, NAME, "weights");

        Dispatch::require_storage(dy, NAME, "dy");
        Dispatch::require_storage(weights, NAME, "weights");
        Dispatch::require_storage(dx, NAME, "dx");
        Dispatch::require_device_dtype(dy, NAME, "dy");
        Dispatch::require_device_dtype(weights, NAME, "weights");
        Dispatch::require_device_dtype(dx, NAME, "dx");

        // A gather reads its inputs while it writes, so the output may share no byte with them.
        Dispatch::require_operand(!dx.region().overlaps(dy.region()), NAME, "dx",
                                  "overlaps 'dy' in memory");
        Dispatch::require_operand(!dx.region().overlaps(weights.region()), NAME, "dx",
                                  "overlaps 'weights' in memory");

        const void* read_dy = dy.data_or_null();
        const void* read_weights = weights.data_or_null();
        void* write_dx = dx.data_or_null();

        OperandSet operands;
        operands.reads(dy).reads(weights).writes(dx);

        TaskDescription description;
        description.name = NAME;
        description.operation = "spatial.window_dot_backward"_op;

        return Internal::execute_typed_task<Dispatch::Domain::Real>(
            recorder_, description, operands, dy.dtype, "Spatial operation",
            [=](auto scalar_type, const Internal::KernelPairDispatcher& dispatcher,
                sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);

                auto source = static_cast<const T*>(read_dy);
                auto tap_weights = static_cast<const T*>(read_weights);
                auto destination = static_cast<T*>(write_dx);

                return dispatcher.run(q, deps,
                    [=]
                    {
                        Kernels::Reference::window_dot_backward<T>(span, taps, source, tap_weights,
                                                                   destination);
                    },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                    {
                        return Kernels::SYCL::window_dot_backward<T>(q2, span, taps, source,
                                                                     tap_weights, destination,
                                                                     deps2);
                    });
            });
    }
} // namespace SushiBLAS
