/**************************************************************************/
/* pool_max_backward.cpp                                                  */
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
 * @file pool_max_backward.cpp
 * @brief Defines SpatialOps::pool_max_backward, which records the gradient of max pooling.
 * @author Mustafa Garip
 */

#include <vector>
#include <cstdint>
#include <SushiBLAS/engine/task.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/engine/spatial.hpp>
#include <SushiBLAS/engine/spatial/window.hpp>
#include "kernels/spatial.hpp"

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    TaskHandle SpatialOps::pool_max_backward(const Tensor& dy, const Tensor& indices, Tensor& dx,
                                             const Spatial::ConvGeometry& g)
    {
        static constexpr const char* NAME = "spatial.pool_max_backward";

        SB_THROW_IF(dy.dtype != dx.dtype,
                    "Spatial operation '{}' has a {} output gradient and a {} input gradient.",
                    NAME, Core::to_string(dy.dtype), Core::to_string(dx.dtype));
        SB_THROW_IF(indices.dtype != Core::DataType::INT64,
                    "Spatial operation '{}' needs an INT64 index tensor; got {}.",
                    NAME, Core::to_string(indices.dtype));

        // dx plays the input role and dy the output role, as in the forward; see src/README.md.
        const Spatial::WindowSpan span = Spatial::pooling_span_for(dx, g, dy, NAME);

        // The kernels address indices with dy's strides, so both must be dense and alike.
        Dispatch::require_same_shape(indices, dy, NAME, "indices", "dy");
        Dispatch::require_contiguous(dy, NAME, "dy");
        Dispatch::require_contiguous(indices, NAME, "indices");
        if (indices.layout != dy.layout)
            Dispatch::Detail::throw_operand_error(NAME, "indices",
                                                  "has a different layout from 'dy'");

        const void* read_dy = dy.data_or_null();
        const void* read_indices = indices.data_or_null();
        void* write_dx = dx.data_or_null();

        OperandSet operands;
        operands.reads(dy).reads(indices).writes(dx);

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = NAME;
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "spatial.pool_max_backward"_op;

        const auto* index_in = static_cast<const int64_t*>(read_indices);

        return Internal::execute_typed_task<Dispatch::Domain::REAL>(
            recorder_, meta, operands, dy.dtype, "Spatial operation",
            [=](auto scalar_type, const Internal::KernelPairDispatcher& dispatcher,
                sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);

                auto source = static_cast<const T*>(read_dy);
                auto destination = static_cast<T*>(write_dx);

                return dispatcher.run(q, deps,
                    [=] { Kernels::Reference::pool_max_backward<T>(span, source, index_in, destination); },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                    {
                        return Kernels::SYCL::pool_max_backward<T>(q2, span, source, index_in,
                                                                   destination, deps2);
                    });
            });
    }
} // namespace SushiBLAS
