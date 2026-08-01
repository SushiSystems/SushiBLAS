/**************************************************************************/
/* red_internal.hpp                                                       */
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
#include <SushiBLAS/core/logger.hpp>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiBLAS 
{
    using namespace SushiRuntime::Graph::Literals;

    namespace Internal 
    {
        /**
         * @brief Helper to execute a tensor reduction operation.
         * 
         * @tparam Dispatcher A generic callable (e.g., a templated lambda) that handles 
         *                    the type-specific SYCL command group and parallel_for. 
         *                    It should have the signature:
         *                    auto operator()(sycl::queue& q, const auto* pT, auto* pR, int64_t size, const std::vector<sycl::event>& deps)
         */
        template<typename Dispatcher>
        sycl::event execute_reduction(Engine& engine, const Tensor& t, Tensor& result, const char* name, SushiRuntime::Graph::OpID op_id, Dispatcher&& dispatch) 
        {
            SB_THROW_IF(result.num_elements != 1, "Result tensor for reduction '{}' must be a scalar (1 element).", name);
            SB_THROW_IF(t.dtype != result.dtype, "Data types must match for reduction operations.");

            int64_t size = t.num_elements;
            void* read_T = t.data();
            void* write_R = result.data();

            std::vector<void*> reads = {read_T};
            std::vector<void*> writes = {write_R};

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
                    SB_THROW_IF(true, "Unsupported data type for reduction operation.");
                    break;
            }
            return sycl::event();
        }

    } // namespace Internal
} // namespace SushiBLAS
