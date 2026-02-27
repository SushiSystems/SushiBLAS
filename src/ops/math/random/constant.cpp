/**************************************************************************/
/* constant.cpp                                                           */
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

#include <complex>
#include <sycl/sycl.hpp>
#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/ops/math/random.hpp>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiBLAS 
{
    using namespace SushiRuntime::Graph::Literals;

    sycl::event RandomOps::constant(Tensor& t, double value) 
    {
        auto op_id = "random.constant"_op;
        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = "random.constant";
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = op_id;
        meta.set_param(0, value);

        const int64_t size = t.num_elements;
        void* ptr = t.storage ? t.storage->data_ptr : nullptr;
        std::vector<void*> reads = {};
        std::vector<void*> writes = {};
        if (ptr) writes.push_back(ptr);

        switch (t.dtype)
        {
            case Core::DataType::HALF:
            {
                engine_.get_graph().add_task(meta, reads, writes,
                    [size, value, pT = t.data_as<sycl::half>()](sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event 
                    {
                        SB_LOG_INFO("RandomOps: constant ({} elements, value: {:.4f})", size, value);
                        return q.fill(pT, static_cast<sycl::half>(value), size, deps);
                    }
                );
                break;
            }
            case Core::DataType::FLOAT32:
            {
                engine_.get_graph().add_task(meta, reads, writes,
                    [size, value, pT = t.data_as<float>()](sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event 
                    {
                        SB_LOG_INFO("RandomOps: constant ({} elements, value: {:.4f})", size, value);
                        return q.fill(pT, static_cast<float>(value), size, deps);
                    }
                );
                break;
            }
            case Core::DataType::FLOAT64:
            {
                engine_.get_graph().add_task(meta, reads, writes,
                    [size, value, pT = t.data_as<double>()](sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event 
                    {
                        SB_LOG_INFO("RandomOps: constant ({} elements, value: {:.4f})", size, value);
                        return q.fill(pT, value, size, deps);
                    }
                );
                break;
            }
            case Core::DataType::COMPLEX32:
            {
                engine_.get_graph().add_task(meta, reads, writes,
                    [size, value, pT = t.data_as<std::complex<float>>()](sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event 
                    {
                        SB_LOG_INFO("RandomOps: constant complex32 ({} elements, value: {:.4f})", size, value);
                        return q.fill(pT, std::complex<float>(static_cast<float>(value), 0.0f), size, deps);
                    }
                );
                break;
            }
            case Core::DataType::COMPLEX64:
            {
                engine_.get_graph().add_task(meta, reads, writes,
                    [size, value, pT = t.data_as<std::complex<double>>()](sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event 
                    {
                        SB_LOG_INFO("RandomOps: constant complex64 ({} elements, value: {:.4f})", size, value);
                        return q.fill(pT, std::complex<double>(value, 0.0), size, deps);
                    }
                );
                break;
            }
            default:
                SB_THROW_IF(true, "Unsupported data type for constant operation.");
        }

        return sycl::event();
    }
} // namespace SushiBLAS
