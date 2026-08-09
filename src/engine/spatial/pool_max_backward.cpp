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
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

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

    sycl::event SpatialOps::pool_max_backward(const Tensor& dy, const Tensor& indices, Tensor& dx,
                                              const Spatial::ConvGeometry& g)
    {
        static constexpr const char* NAME = "spatial.pool_max_backward";

        SB_THROW_IF(dy.dtype != dx.dtype,
                    "Spatial operation '{}' has a {} output gradient and a {} input gradient.",
                    NAME, Core::to_string(dy.dtype), Core::to_string(dx.dtype));
        SB_THROW_IF(indices.dtype != Core::DataType::INT64,
                    "Spatial operation '{}' needs an INT64 index tensor; got {}.",
                    NAME, Core::to_string(indices.dtype));

        // `dx` plays the input role in the span - it has the image's H, W and
        // strides - and `dy` the output role, which is the same assignment the
        // forward made. The geometry must therefore be the forward's, verbatim.
        const Spatial::WindowSpan span = Spatial::window_span_for(dx, g, dy, NAME);

        SB_THROW_IF(indices.num_elements != dy.num_elements,
                    "Spatial operation '{}' needs an index tensor shaped like its output gradient "
                    "({} elements); got {}.",
                    NAME, dy.num_elements, indices.num_elements);

        void* read_dy = dy.data_or_null();
        void* read_indices = indices.data_or_null();
        void* write_dx = dx.data_or_null();

        std::vector<SushiRuntime::Core::ResourceRegion> reads = {};
        if (read_dy) reads.push_back(dy.region());
        if (read_indices) reads.push_back(indices.region());
        std::vector<SushiRuntime::Core::ResourceRegion> writes = {};
        if (write_dx) writes.push_back(dx.region());

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = NAME;
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "spatial.pool_max_backward"_op;

        const auto mode = engine_.get_mode();

        const auto* index_in = static_cast<const int64_t*>(read_indices);

        return Internal::execute_typed_task<Dispatch::Domain::REAL>(
            engine_, meta, reads, writes, dy.dtype, "Spatial operation",
            [=](auto scalar_type, sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
            {
                using T = decltype(scalar_type);

                auto source = static_cast<const T*>(read_dy);
                auto destination = static_cast<T*>(write_dx);

                return Internal::dispatch_with_mode(mode, q, deps,
                    [=] { Kernels::Reference::pool_max_backward<T>(span, source, index_in, destination); },
                    [=](sycl::queue& q2, const std::vector<sycl::event>& deps2) -> sycl::event
                    {
                        return Kernels::SYCL::pool_max_backward<T>(q2, span, source, index_in,
                                                                   destination, deps2);
                    });
            });
    }
} // namespace SushiBLAS
