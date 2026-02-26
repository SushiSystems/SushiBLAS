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
#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/ops/math/random.hpp>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiBLAS 
{
    using namespace SushiRuntime::Graph::Literals;

    sycl::event RandomOps::constant(Tensor& t, double value) 
    {
        const int64_t size = t.num_elements;
        void* write_ptr = t.storage ? t.storage->data_ptr : nullptr;
        std::vector<void*> writes = {};
        if (write_ptr) writes.push_back(write_ptr);

        SushiRuntime::Graph::TaskMetadata meta;
        meta.name = "constant_fill";
        meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
        meta.op_id = "random.constant"_op;
        meta.set_param(0, value);

        SB_LOG_DEBUG("Dispatching Constant Fill Task, Value: {}, Size: {}", value, size);

        engine_.get_graph().add_task(meta, {}, writes,
            [size, value, t_dtype = t.dtype, 
             pH = t.data_as<sycl::half>(), 
             pF = t.data_as<float>(), 
             pD = t.data_as<double>(), 
             pCF = t.data_as<std::complex<float>>(), 
             pCD = t.data_as<std::complex<double>>()]
            (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event 
            {
                switch (t_dtype) 
                {
                    case Core::DataType::HALF:
                        return q.fill(pH, static_cast<sycl::half>(value), size, deps);
                    case Core::DataType::FLOAT32:
                        return q.fill(pF, static_cast<float>(value), size, deps);
                    case Core::DataType::FLOAT64:
                        return q.fill(pD, value, size, deps);
                    case Core::DataType::COMPLEX32:
                        return q.fill(pCF, std::complex<float>(static_cast<float>(value), 0.0f), size, deps);
                    case Core::DataType::COMPLEX64:
                        return q.fill(pCD, std::complex<double>(value, 0.0), size, deps);
                    default:
                        SB_THROW_IF(true, "Unsupported data type for random.constant");
                        return sycl::event();
                }
            }
        );

        return sycl::event();
    }
}
