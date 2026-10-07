/**************************************************************************/
/* window_dot_weights_backward.cpp                                        */
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
 * @file window_dot_weights_backward.cpp
 * @brief Defines SpatialOps::window_dot_weights_backward, which records the weight gradient of
 * the per-channel weighted window sum.
 * @author Mustafa Garip
 */

#include <vector>
#include <cstddef>
#include <SushiBLAS/engine/task.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/engine/spatial.hpp>
#include <SushiBLAS/engine/spatial/window.hpp>
#include <SushiBLAS/support/reduce_axis.hpp>
#include <SushiBLAS/support/device_policy.hpp>
#include "kernels/spatial.hpp"

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    TaskHandle SpatialOps::window_dot_weights_backward(const Tensor& dy, const Tensor& in,
                                                       Tensor& dweights,
                                                       const Spatial::ConvGeometry& g)
    {
        static constexpr const char* NAME = "spatial.window_dot_weights_backward";

        Dispatch::require_dtype(in, dy.dtype, NAME, "in");
        Dispatch::require_dtype(dweights, dy.dtype, NAME, "dweights");

        // in plays the input role and dy the output role, as in the forward; see src/README.md.
        const Spatial::WindowSpan span = Spatial::window_span_for(in, g, dy, NAME);
        const Spatial::TapSpan taps = Spatial::tap_span_for(dweights, span, NAME, "dweights");

        Dispatch::require_storage(dy, NAME, "dy");
        Dispatch::require_storage(in, NAME, "in");
        Dispatch::require_storage(dweights, NAME, "dweights");
        Dispatch::require_device_dtype(dy, NAME, "dy");
        Dispatch::require_device_dtype(in, NAME, "in");
        Dispatch::require_device_dtype(dweights, NAME, "dweights");

        // A fold reads its inputs while it writes, so the output may share no byte with them.
        Dispatch::require_operand(!dweights.region().overlaps(dy.region()), NAME, "dweights",
                                  "overlaps 'dy' in memory");
        Dispatch::require_operand(!dweights.region().overlaps(in.region()), NAME, "dweights",
                                  "overlaps 'in' in memory");

        const void* read_dy = dy.data_or_null();
        const void* read_in = in.data_or_null();
        void* write_dweights = dweights.data_or_null();

        OperandSet operands;
        operands.reads(dy).reads(in).writes(dweights);

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = NAME;
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "spatial.window_dot_weights_backward"_op;

        return Internal::execute_typed_task<Dispatch::Domain::REAL>(
            recorder_, meta, operands, dy.dtype, "Spatial operation",
            [=](auto scalar_type, const Internal::KernelPairDispatcher& dispatcher,
                sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);

                auto gradient = static_cast<const T*>(read_dy);
                auto source = static_cast<const T*>(read_in);
                auto destination = static_cast<T*>(write_dweights);

                return dispatcher.run(q, deps,
                    [=]
                    {
                        Kernels::Reference::window_dot_weights_backward<T>(span, taps, gradient,
                                                                           source, destination);
                    },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                    {
                        // Planned at execution time, from the device that runs the task.
                        const Reduce::AxisPlan geometry = Reduce::plan_axis(
                            Device::profile_for(q2),
                            static_cast<std::size_t>(span.patch_elements()),
                            static_cast<std::size_t>(span.windows()));

                        return Kernels::SYCL::window_dot_weights_backward<T>(
                            q2, span, taps, geometry, gradient, source, destination, deps2);
                    });
            });
    }
} // namespace SushiBLAS
