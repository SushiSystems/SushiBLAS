/**************************************************************************/
/* red_internal.hpp                                                       */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present  Mustafa Garip & Sushi Systems              */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
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
