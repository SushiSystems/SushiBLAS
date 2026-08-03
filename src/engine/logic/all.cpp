/**************************************************************************/
/* all.cpp                                                                */
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
#include <sycl/sycl.hpp>
#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/engine/logic/logic.hpp>
#include <SushiRuntime/core/resource_region.hpp>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiBLAS 
{
    using namespace SushiRuntime::Graph::Literals;

    namespace
    {
        // On sycl::reduction deliberately, and safe there in a way a floating-point sum is
        // not. The spec leaves a reduction's combination order unspecified, which only
        // matters for a combiner whose result depends on it. This one reduces over exactly
        // two values, T(0) and T(1), under `minimum`, which is associative, commutative and
        // exact on that domain - every order gives the same bits. Contrast `+` on floats,
        // where the order is visible in the result and free to shift with the launch
        // configuration.
        template<typename T>
        sycl::event all_dispatch(sycl::queue& queue, const T* pT, T* pR, int64_t size, const std::vector<sycl::event>& deps)
        {
            SB_LOG_INFO("Logic ALL: {} elements", size);
            return queue.submit([&](sycl::handler& h) 
            {
                h.depends_on(deps);
                h.parallel_for(sycl::range<1>(size), sycl::reduction(pR, T(1), sycl::minimum<T>()), [=](sycl::id<1> idx, auto& reducer) 
                {
                    reducer.combine((pT[idx[0]] != T(0)) ? T(1) : T(0));
                });
            });
        }
    }

    sycl::event LogicOps::all(const Tensor& t, Tensor& result) 
    {
        SB_THROW_IF(result.num_elements != 1, "Result tensor for 'all' must be a scalar (1 element).");
        SB_THROW_IF(t.dtype != result.dtype, "Data types must match for logic operations.");

        int64_t size = t.num_elements;
        void* read_T = t.data_or_null();
        void* write_R = result.data_or_null();

        std::vector<SushiRuntime::Core::ResourceRegion> reads = {};
        if (read_T) reads.push_back(t.region());
        std::vector<SushiRuntime::Core::ResourceRegion> writes = {};
        if (write_R) writes.push_back(result.region());

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = "logic.all";
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "logic.all"_op;

        switch (t.dtype)
        {
            case Core::DataType::HALF:
                engine_.get_graph().add_task(meta, reads, writes,
                    [size, pT=t.data_as<sycl::half>(), pR=result.data_as<sycl::half>()]
                    (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event 
                    {
                        return all_dispatch<sycl::half>(q, pT, pR, size, deps);
                    });
                break;
            case Core::DataType::FLOAT32:
                engine_.get_graph().add_task(meta, reads, writes,
                    [size, pT=t.data_as<float>(), pR=result.data_as<float>()]
                    (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event 
                    {
                        return all_dispatch<float>(q, pT, pR, size, deps);
                    });
                break;
            case Core::DataType::FLOAT64:
                engine_.get_graph().add_task(meta, reads, writes,
                    [size, pT=t.data_as<double>(), pR=result.data_as<double>()]
                    (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event 
                    {
                        return all_dispatch<double>(q, pT, pR, size, deps);
                    });
                break;
            default:
                SB_THROW_IF(true, "Unsupported data type for logic operation.");
        }
        return sycl::event();
    }
} // namespace SushiBLAS
