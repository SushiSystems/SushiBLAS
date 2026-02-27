/**************************************************************************/
/* level1_internal.hpp                                                    */
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
#include <complex>
#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>
#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/tensor.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiBLAS 
{
    namespace Internal 
    {
        template<typename T> struct is_complex : std::false_type {};
        template<typename T> struct is_complex<std::complex<T>> : std::true_type {};
        template<typename T> inline constexpr bool is_complex_v = is_complex<T>::value;

        /**
         * @brief Internal helper to execute a generically dispatched MKL level 1 BLAS task.
         */
        template<typename Func>
        sycl::event execute_level1(Engine& engine, const char* name, SushiRuntime::Graph::OpID op_id, Core::DataType dtype, const std::vector<void*>& reads, const std::vector<void*>& writes, const std::vector<float>& params, Func&& op_func)
        {
            SushiRuntime::Graph::TaskMetadata meta;
            meta.name = name;
            meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
            meta.op_id = op_id;
            for (size_t i = 0; i < params.size(); ++i) meta.set_param(i, params[i]);

            engine.get_graph().add_task(meta, reads, writes,
                [dtype, op_func, name](sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event 
                {
                    SB_LOG_INFO("MKL Level 1 {}: starting", name);
                    switch (dtype) 
                    {
                        case Core::DataType::HALF:
                            return op_func(sycl::half{}, q, deps);
                        case Core::DataType::FLOAT32:
                            return op_func(float{}, q, deps);
                        case Core::DataType::FLOAT64:
                            return op_func(double{}, q, deps);
                        case Core::DataType::COMPLEX32:
                            return op_func(std::complex<float>{}, q, deps);
                        case Core::DataType::COMPLEX64:
                            return op_func(std::complex<double>{}, q, deps);
                        default: 
                            SB_LOG_ERROR("Unsupported data type for {}", name);
                            break;
                    }
                    return sycl::event();
                });
            return sycl::event();
        }

    } // namespace Internal
} // namespace SushiBLAS
