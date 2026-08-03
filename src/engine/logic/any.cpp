/**************************************************************************/
/* any.cpp                                                                */
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
#include <SushiBLAS/engine/dispatch.hpp>
#include <SushiBLAS/engine/logic/logic.hpp>
#include <SushiRuntime/core/resource_region.hpp>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    namespace
    {
        // On sycl::reduction deliberately - see the note in all.cpp, which applies here
        // with `maximum` in place of `minimum`. The combiner sees only T(0) and T(1), on
        // which max is associative, commutative and exact, so the combination order the
        // spec leaves free cannot change a single bit of the answer.
        template<typename T>
        sycl::event any_dispatch(sycl::queue& queue, const T* pT, T* pR, int64_t size, const std::vector<sycl::event>& deps)
        {
            SB_LOG_INFO("Logic ANY: {} elements", size);
            return queue.submit([&](sycl::handler& h)
            {
                h.depends_on(deps);
                // `initialize_to_identity`, as in all.cpp and for the same reason: without
                // it a SYCL reduction folds whatever the result tensor already holds into
                // the answer, so `any` over a tensor of zeros returned 1 whenever the
                // caller reused a result that had been 1 before.
                h.parallel_for(sycl::range<1>(size),
                               sycl::reduction(pR, T(0), sycl::maximum<T>(),
                                               sycl::property::reduction::initialize_to_identity{}),
                               [=](sycl::id<1> idx, auto& reducer)
                {
                    reducer.combine((pT[idx[0]] != T(0)) ? T(1) : T(0));
                });
            });
        }
    }

    sycl::event LogicOps::any(const Tensor& t, Tensor& result)
    {
        SB_THROW_IF(result.num_elements != 1, "Result tensor for 'any' must be a scalar (1 element).");
        SB_THROW_IF(t.dtype != result.dtype, "Data types must match for logic operations.");

        int64_t size = t.num_elements;
        void* read_T = t.data_or_null();
        void* write_R = result.data_or_null();

        std::vector<SushiRuntime::Core::ResourceRegion> reads = {};
        if (read_T) reads.push_back(t.region());
        std::vector<SushiRuntime::Core::ResourceRegion> writes = {};
        if (write_R) writes.push_back(result.region());

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = "logic.any";
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "logic.any"_op;

        // Domain::REAL_AND_INTEGRAL, not REAL: a truth test is exactly as meaningful on an
        // integer tensor as on a float one, and an integer element is the honest way to
        // carry the 1/0 this family produces. Complex stays out - `logic_not_zero` would
        // work, but `sycl::minimum`/`sycl::maximum` over std::complex does not, and a
        // reduction needs an ordering.
        //
        // The dtype switch itself is Dispatch::with_element_type, the same one the logic
        // dispatch seam uses, so the arms and the by-name rejection are not written twice.
        Dispatch::with_element_type<Dispatch::Domain::REAL_AND_INTEGRAL>(
            t.dtype, "Logic operation", meta.name, [&](auto tag)
        {
            using T = typename decltype(tag)::type;

            engine_.get_graph().add_task(meta, reads, writes,
                [size, pT = t.data_as<T>(), pR = result.data_as<T>()]
                (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
                {
                    return any_dispatch<T>(q, pT, pR, size, deps);
                });
        });
        return sycl::event();
    }
} // namespace SushiBLAS
