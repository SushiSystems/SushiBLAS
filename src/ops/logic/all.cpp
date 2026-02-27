/**************************************************************************/
/* all.cpp                                                                */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present  Mustafa Garip & Sushi Systems              */
/*                                                                   	  */
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

#include <vector>
#include <sycl/sycl.hpp>
#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/ops/logic/logic.hpp>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiBLAS 
{
    using namespace SushiRuntime::Graph::Literals;

    namespace
    {
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
        void* read_T = t.storage ? t.storage->data_ptr : nullptr;
        void* write_R = result.storage ? result.storage->data_ptr : nullptr;

        std::vector<void*> reads = {};
        if (read_T) reads.push_back(read_T);
        std::vector<void*> writes = {};
        if (write_R) writes.push_back(write_R);

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
