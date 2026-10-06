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
#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/engine/task.hpp>
#include <SushiBLAS/engine/spatial.hpp>
#include <SushiBLAS/engine/spatial/window.hpp>
#include "kernels/spatial.hpp"

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    sycl::event SpatialOps::pool_max(const Tensor& in, Tensor& out, Tensor& indices,
                                     const Spatial::ConvGeometry& g)
    {
        static constexpr const char* NAME = "spatial.pool_max";

        SB_THROW_IF(in.dtype != out.dtype,
                    "Spatial operation '{}' has a {} input and a {} output.",
                    NAME, Core::to_string(in.dtype), Core::to_string(out.dtype));

        // The argmax buffer must be INT64; see src/README.md.
        SB_THROW_IF(indices.dtype != Core::DataType::INT64,
                    "Spatial operation '{}' needs an INT64 index tensor; got {}.",
                    NAME, Core::to_string(indices.dtype));

        const Spatial::WindowSpan span = Spatial::window_span_for(in, g, out, NAME);

        SB_THROW_IF(indices.rank != out.rank || indices.num_elements != out.num_elements,
                    "Spatial operation '{}' needs an index tensor shaped like its output "
                    "({} elements); got {}.",
                    NAME, out.num_elements, indices.num_elements);

        void* read_in = in.data_or_null();
        void* write_out = out.data_or_null();
        void* write_indices = indices.data_or_null();

        std::vector<SushiRuntime::Core::ResourceRegion> reads = {};
        if (read_in) reads.push_back(in.region());
        std::vector<SushiRuntime::Core::ResourceRegion> writes = {};
        if (write_out) writes.push_back(out.region());
        if (write_indices) writes.push_back(indices.region());

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = NAME;
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "spatial.pool_max"_op;

        const auto mode = engine_.get_mode();

        // The switch runs on the value dtype; the index pointer is int64_t*; see src/README.md.
        auto* index_out = static_cast<int64_t*>(write_indices);

        return Internal::execute_typed_task<Dispatch::Domain::REAL>(
            engine_, meta, reads, writes, in.dtype, "Spatial operation",
            [=](auto scalar_type, sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);

                auto source = static_cast<const T*>(read_in);
                auto destination = static_cast<T*>(write_out);

                return Internal::dispatch_with_mode(mode, q, deps,
                    [=] { Kernels::Reference::pool_max<T>(span, source, destination, index_out); },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                    {
                        return Kernels::SYCL::pool_max<T>(q2, span, source, destination,
                                                          index_out, deps2);
                    });
            });
    }
} // namespace SushiBLAS
