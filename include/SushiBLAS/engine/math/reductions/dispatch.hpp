/**************************************************************************/
/* dispatch.hpp                                                           */
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

#include <vector>
#include <sycl/sycl.hpp>
#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/tensor.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/engine/dispatch.hpp>
#include <SushiRuntime/core/resource_region.hpp>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiBLAS
{
    namespace Dispatch
    {
        /**
         * @brief Records a whole-tensor reduction to a scalar result tensor.
         *
         * The generic seam behind every `ReductionOps` entry point. Unlike the elementwise
         * and logic families, the caller supplies the entire command group rather than a
         * per-element functor: a reduction's shape (sycl::reduction, a tree pass, or a
         * sequential `single_task` for the numerically delicate ones) is the caller's
         * choice, and only the graph bookkeeping and the dtype switch are shared.
         *
         * @tparam Dispatcher The command-group builder type, deduced.
         * @param engine The engine whose task graph the op is recorded on.
         * @param t The tensor to reduce.
         * @param result The scalar (single-element) output tensor; must share `t`'s dtype.
         * @param name Human-readable op name, used for logging and failure messages.
         * @param op_id Stable operation identifier for the task graph.
         * @param dispatch A generic callable (typically a templated lambda) with the
         *        signature `(sycl::queue& q, const auto* pT, auto* pR, int64_t size,
         *        const std::vector<sycl::event>& deps) -> sycl::event`, instantiated once
         *        per supported element type.
         * @return An empty sycl::event; the work is deferred until `Engine::execute()`.
         *
         * @warning This is a low-level generic seam. Whatever `dispatch` submits runs on the
         *          device, so any lambda it nests must be device-copyable, capture by value,
         *          hold no host pointers, allocate nothing and throw nothing. `dispatch`
         *          itself runs on the host at execution time and owns the `depends_on(deps)`
         *          call - dropping it silently breaks the graph's ordering guarantees.
         * @throws std::runtime_error if `result` is not scalar, if the dtypes disagree, or if
         *         the dtype is one reductions do not support (HALF, FLOAT32 and FLOAT64 are;
         *         the complex types are not, since `sycl::reduction`'s operators need an
         *         ordering and an identity that `std::complex` does not supply).
         */
        template<typename Dispatcher>
        sycl::event execute_reduction(Engine& engine, const Tensor& t, Tensor& result, const char* name, SushiRuntime::Graph::OpID op_id, Dispatcher&& dispatch)
        {
            SB_THROW_IF(result.num_elements != 1, "Result tensor for reduction '{}' must be a scalar (1 element).", name);
            SB_THROW_IF(t.dtype != result.dtype, "Data types must match for reduction operations.");

            int64_t size = t.num_elements;
            void* read_T = t.data();
            void* write_R = result.data();

            std::vector<SushiRuntime::Core::ResourceRegion> reads = {read_T};
            std::vector<SushiRuntime::Core::ResourceRegion> writes = {write_R};

            SushiRuntime::Graph::TaskMetadata meta;
            meta.name = name;
            meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
            meta.op_id = op_id;

            switch (t.dtype)
            {
                case Core::DataType::HALF:
                    engine.get_graph().add_task(meta, reads, writes,
                        [size, pT=(sycl::half*)read_T, pR=(sycl::half*)write_R, dispatch, name]
                        (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
                        {
                            SB_LOG_INFO("Reduction {}: {} elements", name, size);
                            return dispatch(q, pT, pR, size, deps);
                        });
                    break;
                case Core::DataType::FLOAT32:
                    engine.get_graph().add_task(meta, reads, writes,
                        [size, pT=(float*)read_T, pR=(float*)write_R, dispatch, name]
                        (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
                        {
                            SB_LOG_INFO("Reduction {}: {} elements", name, size);
                            return dispatch(q, pT, pR, size, deps);
                        });
                    break;
                case Core::DataType::FLOAT64:
                    engine.get_graph().add_task(meta, reads, writes,
                        [size, pT=(double*)read_T, pR=(double*)write_R, dispatch, name]
                        (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
                        {
                            SB_LOG_INFO("Reduction {}: {} elements", name, size);
                            return dispatch(q, pT, pR, size, deps);
                        });
                    break;
                default:
                    SB_THROW_IF(true, "Reduction '{}' does not support dtype {}.", name, Core::to_string(t.dtype));
                    break;
            }
            return sycl::event();
        }

    } // namespace Dispatch
} // namespace SushiBLAS
