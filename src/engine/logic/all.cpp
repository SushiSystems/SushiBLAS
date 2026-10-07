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
 * @file all.cpp
 * @brief Defines LogicOps::all, which records the reduction that tests whether every element is
 * non-zero.
 * @author Mustafa Garip
 */

#include <vector>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/engine/dispatch.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/engine/logic/logic.hpp>
#include <SushiRuntime/core/resource_region.hpp>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    namespace
    {
        // minimum over T(0) and T(1) is exact in any order; see src/README.md.
        template<typename T>
        sycl::event all_dispatch(sycl::queue& queue, const T* pT, T* pR, int64_t size, const std::vector<sycl::event>& deps)
        {
            SB_LOG_INFO("Logic ALL: {} elements", size);
            if (size == 0)
            {
                return queue.submit([&](sycl::handler& h)
                {
                    h.depends_on(deps);
                    h.single_task([=]() { pR[0] = T(1); });
                });
            }

            return queue.submit([&](sycl::handler& h)
            {
                h.depends_on(deps);
                // The reduction starts from its identity, not from the result; see src/README.md.
                h.parallel_for(sycl::range<1>(size),
                               sycl::reduction(pR, T(1), sycl::minimum<T>(),
                                               sycl::property::reduction::initialize_to_identity{}),
                               [=](sycl::id<1> idx, auto& reducer)
                {
                    reducer.combine((pT[idx[0]] != T(0)) ? T(1) : T(0));
                });
            });
        }
    }

    TaskHandle LogicOps::all(const Tensor& t, Tensor& result)
    {
        recorder_.require_sycl_path("Logic operation", "logic.all");
        Dispatch::require_storage(t, "logic.all", "t");
        Dispatch::require_device_dtype(t, "logic.all", "t");
        Dispatch::require_storage(result, "logic.all", "result");
        Dispatch::require_device_dtype(result, "logic.all", "result");
        Dispatch::require_contiguous(t, "logic.all", "t");
        Dispatch::require_scalar(result, "logic.all", "result");
        Dispatch::require_dtype(result, t.dtype, "logic.all", "result");

        int64_t size = t.num_elements;
        const void* read_T = t.data_or_null();

        OperandSet operands;
        operands.reads(t).writes(result);

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = "logic.all";
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "logic.all"_op;

        TaskHandle handle;

        // A truth test serves real and integer dtypes and refuses complex; see src/README.md.
        Dispatch::require_domain<Dispatch::Domain::REAL_AND_INTEGRAL>(t.dtype, "Logic operation",
                                                                      "logic.all");
        Dispatch::with_element_type<Dispatch::Domain::REAL_AND_INTEGRAL>(
            t.dtype, "Logic operation", meta.name, [&](auto tag)
        {
            using T = typename decltype(tag)::type;

            handle = recorder_.record(meta, operands, "Logic operation",
                [size, pT = static_cast<const T*>(read_T), pR = result.data_as<T>()]
                (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
                {
                    return all_dispatch<T>(q, pT, pR, size, deps);
                });
        });
        return handle;
    }
} // namespace SushiBLAS
