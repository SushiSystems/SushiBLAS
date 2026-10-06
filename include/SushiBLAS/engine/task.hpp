/**************************************************************************/
/* task.hpp                                                               */
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
 * @file task.hpp
 * @brief Declares the recorder of one dtype-dispatched task and the execution-mode dispatcher.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <complex>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/core/traits.hpp>
#include <SushiBLAS/engine/dispatch.hpp>
#include <SushiRuntime/core/resource_region.hpp>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiBLAS
{
    namespace Internal
    {
        /**
         * @brief Records one dtype-dispatched task on the engine's graph.
         * @param family What kind of operation this is, for diagnostics.
         * @param op_func Callable taking a scalar of the element type, the queue and the dependencies.
         * @throws std::runtime_error at record time if dtype falls outside Dom.
         * @see include/SushiBLAS/engine/README.md
         */
        template<Dispatch::Domain Dom = Dispatch::Domain::REAL_AND_COMPLEX, typename Func>
        sycl::event execute_typed_task(Engine& engine, const SushiRuntime::Graph::TaskMetadata& meta,
                                       const std::vector<SushiRuntime::Core::ResourceRegion>& reads,
                                       const std::vector<SushiRuntime::Core::ResourceRegion>& writes,
                                       Core::DataType dtype, const char* family, Func&& op_func)
        {
            Dispatch::require_domain<Dom>(dtype, family, meta.name);

            engine.get_graph().add_task(meta, reads, writes,
                [dtype, op_func, family, name = meta.name](sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
                {
                    // The seven arms mirror Dispatch::with_element_type; see include/SushiBLAS/engine/README.md.
                    switch (dtype)
                    {
                        case Core::DataType::HALF:
                            return op_func(sycl::half{}, q, deps);
                        case Core::DataType::FLOAT32:
                            return op_func(float{}, q, deps);
                        case Core::DataType::FLOAT64:
                            return op_func(double{}, q, deps);
                        case Core::DataType::COMPLEX32:
                            if constexpr (Dispatch::domain_admits(Dom, Dispatch::Domain::COMPLEX))
                                return op_func(std::complex<float>{}, q, deps);
                            break;
                        case Core::DataType::COMPLEX64:
                            if constexpr (Dispatch::domain_admits(Dom, Dispatch::Domain::COMPLEX))
                                return op_func(std::complex<double>{}, q, deps);
                            break;
                        case Core::DataType::INT32:
                            if constexpr (Dispatch::domain_admits(Dom, Dispatch::Domain::INTEGRAL))
                                return op_func(int32_t{}, q, deps);
                            break;
                        case Core::DataType::INT64:
                            if constexpr (Dispatch::domain_admits(Dom, Dispatch::Domain::INTEGRAL))
                                return op_func(int64_t{}, q, deps);
                            break;
                    }

                    // The switch ends in a throw, so a dtype that slips past require_domain cannot no-op; see include/SushiBLAS/engine/README.md.
                    SB_THROW_IF(true, "{} '{}' does not support dtype {}.",
                                family, name, Core::to_string(dtype));
                    return sycl::event();
                });
            return sycl::event();
        }

        /**
         * @brief Dispatches a kernel through the FORCE_REFERENCE, normal or VERIFY execution mode.
         * @param reference_fn Callable `() -> void` running the pure-C++ reference kernel.
         * @param sycl_fn Callable `(sycl::queue&, const std::vector<sycl::event>&) -> sycl::event`.
         * @return The submitted event, or an empty one under FORCE_REFERENCE.
         * @see include/SushiBLAS/engine/README.md
         */
        template<typename ReferenceFn, typename SyclFn>
        sycl::event dispatch_with_mode(ExecutionMode mode, sycl::queue& q, const std::vector<sycl::event>& deps,
                                        ReferenceFn&& reference_fn, SyclFn&& sycl_fn)
        {
            if (mode == ExecutionMode::FORCE_REFERENCE)
            {
                sycl::event::wait(deps);
                reference_fn();
                return sycl::event();
            }

            auto ev = sycl_fn(q, deps);

            if (mode == ExecutionMode::VERIFY)
            {
                ev.wait();
                reference_fn();
            }
            return ev;
        }
    }
}
