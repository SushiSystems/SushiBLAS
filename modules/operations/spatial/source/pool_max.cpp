/**************************************************************************/
/* pool_max.cpp                                                           */
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
 * @file pool_max.cpp
 * @brief Defines SpatialOps::pool_max, which records max pooling with its argmax indices.
 * @author Mustafa Garip
 */

#include <vector>
#include <cstdint>
#include <SushiBLAS/engine/task.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/spatial/spatial.hpp>
#include <SushiBLAS/spatial/window.hpp>
#include <SushiBLAS/spatial_kernels/spatial.hpp>
#include <SushiBLAS/engine/task_description.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Literals;

    TaskHandle SpatialOps::pool_max(const Tensor& in, Tensor& out, Tensor& indices,
                                    const Spatial::ConvGeometry& g)
    {
        static constexpr const char* NAME = "spatial.pool_max";

        Dispatch::require_dtype(out, in.dtype, NAME, "out");

        // The argmax buffer must be INT64; see modules/operations/README.md.
        Dispatch::require_dtype(indices, Core::DataType::Int64, NAME, "indices");

        Dispatch::require_device_dtype(in, NAME, "in");
        Dispatch::require_device_dtype(out, NAME, "out");
        Dispatch::require_device_dtype(indices, NAME, "indices");

        const Spatial::WindowSpan span = Spatial::pooling_span_for(in, g, out, NAME);

        // The kernels address indices with out's strides, so both must be dense and alike.
        Dispatch::require_same_shape(indices, out, NAME, "indices", "out");
        Dispatch::require_contiguous(out, NAME, "out");
        Dispatch::require_contiguous(indices, NAME, "indices");
        if (indices.layout != out.layout)
            Dispatch::Detail::throw_operand_error(NAME, "indices",
                                                  "has a different layout from 'out'");

        const void* read_in = in.data_or_null();
        void* write_out = out.data_or_null();
        void* write_indices = indices.data_or_null();

        OperandSet operands;
        operands.reads(in).writes(out).writes(indices);

        TaskDescription description;
        description.name = NAME;
        description.operation = "spatial.pool_max"_op;
        description.shape = WindowShape{span.output_elements(), g.kernel_h * g.kernel_w, in.dtype};

        // The switch runs on the value dtype; the index pointer is int64_t*; see modules/operations/README.md.
        auto* index_out = static_cast<int64_t*>(write_indices);

        return Internal::execute_typed_task<Dispatch::Domain::Real>(
            recorder_, description, operands, in.dtype, "Spatial operation",
            [=](auto scalar_type, const KernelPairDispatcher& dispatcher,
                sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);

                auto source = static_cast<const T*>(read_in);
                auto destination = static_cast<T*>(write_out);

                return dispatcher.run(q, deps,
                    [=] { Kernels::Reference::pool_max<T>(span, source, destination, index_out); },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                    {
                        return Kernels::SYCL::pool_max<T>(q2, span, source, destination,
                                                          index_out, deps2);
                    });
            });
    }
} // namespace SushiBLAS
