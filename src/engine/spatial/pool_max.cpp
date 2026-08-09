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

    sycl::event SpatialOps::pool_max(const Tensor& in, Tensor& out, Tensor& indices,
                                     const Spatial::ConvGeometry& g)
    {
        static constexpr const char* NAME = "spatial.pool_max";

        SB_THROW_IF(in.dtype != out.dtype,
                    "Spatial operation '{}' has a {} input and a {} output.",
                    NAME, Core::to_string(in.dtype), Core::to_string(out.dtype));

        // An index is not a number. fp32 represents integers exactly only below
        // 2^24, and a 224x224x64 feature map passes that at batch 6 - so the
        // argmax buffer is INT64 and nothing else, checked by name here rather
        // than reinterpreted silently in the kernel.
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

        // The dtype driving the switch is the *value* tensor's; the index
        // pointer is captured as int64_t* outside it, because its type is fixed
        // by the operation rather than by the operands.
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
