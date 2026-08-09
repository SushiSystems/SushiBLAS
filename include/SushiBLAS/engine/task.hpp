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

#pragma once

/**
 * @file task.hpp
 * @brief Recording one dispatched task on the engine's graph, for any domain.
 *
 * This is `engine/blas/utils.hpp`'s `execute_blas_task` with the one thing that
 * was BLAS-specific about it - the word "BLAS" in its diagnostics - lifted into
 * a parameter. Nothing else changed, and `execute_blas_task` still exists and
 * still says "BLAS operation"; it forwards here.
 *
 * The move happened because a second domain arrived. `spatial.im2col` reported
 * as a "BLAS operation" when its dtype is refused sends a reader to the wrong
 * half of the library, and the alternative - a second copy of the seven-arm
 * dtype switch - is seven arms to forget when a `DataType` is added.
 *
 * The include list is `blas/utils.hpp`'s, minus what only the vector helper
 * needed. `<SushiBLAS/engine.hpp>` is included outright rather than
 * forward-declared: both templates below need `Engine` complete (one calls
 * `get_graph()`, the other compares against `ExecutionMode`), and a
 * forward declaration that only works because every includer happens to have
 * pulled `engine.hpp` first is a trap rather than a decoupling.
 */

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
         * @brief Registers one generically-dispatched task on @p engine's graph.
         *
         * One `add_task`, one `switch (dtype)` dispatching a generically-templated
         * closure. An unsupported dtype fails loudly at record time (P1 corollary),
         * never silently no-ops.
         *
         * @tparam Dom Which element types to instantiate @p op_func for, in exactly
         *         the sense `Dispatch::Domain` defines for the math and logic
         *         families. It defaults to REAL_AND_COMPLEX because that is what
         *         every built-in BLAS *arithmetic* op wants; the parameter exists
         *         for callers whose functor is written outside this library and is
         *         therefore no more likely to be well-formed for `std::complex`
         *         than an elementwise lambda is, and for movement ops like `copy`
         *         and the whole spatial family, which relocate elements without
         *         doing arithmetic to them and so pass `Domain::ALL`.
         * @tparam Func The functor type, deduced.
         * @param engine The engine whose task graph the op is recorded on.
         * @param meta The task metadata to record.
         * @param reads Regions the task reads.
         * @param writes Regions the task writes.
         * @param dtype The operands' runtime element type.
         * @param family What kind of operation this is, for diagnostics -
         *        "BLAS operation", "Spatial operation".
         * @param op_func A callable
         *        `(auto scalar_type, sycl::queue&, const std::vector<sycl::event>&) -> sycl::event`,
         *        generic over the runtime dtype via `decltype(scalar_type)`.
         * @return An empty sycl::event; the work is deferred until `Engine::execute()`.
         * @throws std::runtime_error at record time if @p dtype falls outside @p Dom.
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
                    // Spelled out rather than routed through Dispatch::with_element_type,
                    // which is what the elementwise, non-linear and logic families use.
                    // Two reasons, and the second is the binding one. `op_func` takes a
                    // *value* of the element type (`decltype(scalar_type)`) alongside the
                    // queue and the dependency list and returns the submitted event, which
                    // is not the one-argument, void-returning arm that helper calls; and
                    // wrapping it to fit - a nested generic lambda, inside this lambda,
                    // inside this template - crashes this project's clang outright (a
                    // null-dereference during instantiation of the HALF arm, under
                    // -fdelayed-template-parsing). Seven arms here, the
                    // same seven and the same domain rule as with_element_type, kept in step
                    // by hand until that compiler bug is gone.
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

                    // Unreachable in practice - require_domain rejected this dtype at record
                    // time - but a switch that can fall through to nothing is how a silent
                    // no-op gets in, so it ends in a throw rather than in a default return.
                    SB_THROW_IF(true, "{} '{}' does not support dtype {}.",
                                family, name, Core::to_string(dtype));
                    return sycl::event();
                });
            return sycl::event();
        }

        /**
         * @brief Collapses the FORCE_REFERENCE / normal / VERIFY execution-mode triad that
         * ops used to hand-write at every call site (5x in gemm.cpp, 5x in syrk.cpp).
         *
         * @tparam ReferenceFn The reference callable's type, deduced.
         * @tparam SyclFn The device callable's type, deduced.
         * @param mode The engine's current execution mode.
         * @param q The queue to submit to.
         * @param deps Events this submission must depend on.
         * @param reference_fn A callable `() -> void` running the pure-C++ reference kernel.
         * @param sycl_fn A callable `(sycl::queue&, const std::vector<sycl::event>&) -> sycl::event`
         *        running the SYCL/MKL kernel.
         * @return The submitted event, or an empty one under FORCE_REFERENCE.
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
