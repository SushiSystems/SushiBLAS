/**************************************************************************/
/* relu.cpp                                                             */
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

#include <vector>
#include <sycl/sycl.hpp>
#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/ops/math/nonlinear.hpp>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiBLAS 
{
    using namespace SushiRuntime::Graph::Literals;

    namespace
    {
        template<typename T>
        sycl::event relu_forward_dispatch(sycl::queue& queue, T* data, int64_t size, const std::vector<sycl::event>& deps)
        {
            SB_LOG_INFO("ReLU Forward: {} elements", size);
            return queue.submit([&](sycl::handler& h) 
            {
                h.depends_on(deps);
                h.parallel_for(sycl::range<1>(size), [=](sycl::id<1> idx) 
                {
                    data[idx[0]] = data[idx[0]] > T(0) ? data[idx[0]] : T(0);
                });
            });
        }

        template<typename T>
        sycl::event relu_backward_dispatch(sycl::queue& queue, const T* dy, const T* x, T* dx, int64_t size, const std::vector<sycl::event>& deps)
        {
            SB_LOG_INFO("ReLU Backward: {} elements", size);
            return queue.submit([&](sycl::handler& h) 
            {
                h.depends_on(deps);
                h.parallel_for(sycl::range<1>(size), [=](sycl::id<1> idx) 
                {
                    dx[idx[0]] = dy[idx[0]] * (x[idx[0]] > T(0) ? T(1) : T(0));
                });
            });
        }
    } // namespace Anonymous

    sycl::event NonLinearOps::relu(Tensor& t) 
    {
        int64_t size = t.num_elements;
        
        void* write_t = t.storage ? t.storage->data_ptr : nullptr;
        std::vector<void*> reads = {};
        std::vector<void*> writes = {};
        if (write_t) writes.push_back(write_t);

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = "math_relu";
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "math.relu"_op;

        switch (t.dtype)
        {
            // TODO: Add support for Core::DataType::HALF
            case Core::DataType::FLOAT32:
                engine_.get_graph().add_task(meta, reads, writes,
                    [size, pT=t.data_as<float>()]
                    (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event 
                    {
                        return relu_forward_dispatch<float>(q, pT, size, deps);
                    }
                );
                break;
            case Core::DataType::FLOAT64:
                engine_.get_graph().add_task(meta, reads, writes,
                    [size, pT=t.data_as<double>()]
                    (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event 
                    {
                        return relu_forward_dispatch<double>(q, pT, size, deps);
                    }
                );
                break;
            default:
                SB_THROW_IF(true, "Unsupported data type for relu operation.");
        }
        return sycl::event();
    }

    sycl::event NonLinearOps::relu_backward(const Tensor& dy, const Tensor& x, Tensor& dx)
    {
        SB_THROW_IF(dy.num_elements != x.num_elements || dy.num_elements != dx.num_elements, "Tensor sizes must match for relu_backward.");
        SB_THROW_IF(dy.dtype != x.dtype || dy.dtype != dx.dtype, "Data types must match for relu_backward.");

        int64_t size = x.num_elements;

        void* read_dy = dy.storage ? dy.storage->data_ptr : nullptr;
        void* read_x = x.storage ? x.storage->data_ptr : nullptr;
        void* write_dx = dx.storage ? dx.storage->data_ptr : nullptr;

        std::vector<void*> reads = {};
        if (read_dy) reads.push_back(read_dy);
        if (read_x) reads.push_back(read_x);
        std::vector<void*> writes = {};
        if (write_dx) writes.push_back(write_dx);

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = "math_relu_bw";
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "math.relu_backward"_op;

        switch (x.dtype)
        {
            // TODO: Add support for Core::DataType::HALF
            case Core::DataType::FLOAT32:
                engine_.get_graph().add_task(meta, reads, writes,
                    [size, pDY=dy.data_as<float>(), pX=x.data_as<float>(), pDX=dx.data_as<float>()]
                    (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event 
                    {
                        return relu_backward_dispatch<float>(q, pDY, pX, pDX, size, deps);
                    }
                );
                break;
            case Core::DataType::FLOAT64:
                engine_.get_graph().add_task(meta, reads, writes,
                    [size, pDY=dy.data_as<double>(), pX=x.data_as<double>(), pDX=dx.data_as<double>()]
                    (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event 
                    {
                        return relu_backward_dispatch<double>(q, pDY, pX, pDX, size, deps);
                    }
                );
                break;
            default:
                SB_THROW_IF(true, "Unsupported data type for relu_backward operation.");
        }
        return sycl::event();
    }
} // namespace SushiBLAS
