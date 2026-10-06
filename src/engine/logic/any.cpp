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
 * @file any.cpp
 * @brief Defines LogicOps::any, which records the reduction that tests whether any element is non-
 * zero.
 * @author Mustafa Garip
 */

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
        // maximum over T(0) and T(1) is exact in any order; see src/README.md.
        template<typename T>
        sycl::event any_dispatch(sycl::queue& queue, const T* pT, T* pR, int64_t size, const std::vector<sycl::event>& deps)
        {
            SB_LOG_INFO("Logic ANY: {} elements", size);
            return queue.submit([&](sycl::handler& h)
            {
                h.depends_on(deps);
                // The reduction starts from its identity, not from the result; see src/README.md.
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

        // A truth test serves real and integer dtypes and refuses complex; see src/README.md.
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
