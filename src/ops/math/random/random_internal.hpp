/**************************************************************************/
/* random_internal.hpp                                                    */
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

#include <cmath>
#include <vector>
#include <complex>
#include <type_traits>
#include <oneapi/mkl.hpp>
#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/tensor.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiBLAS 
{
    namespace Internal 
    {
        template <typename T>
        struct is_complex : std::false_type {};
        
        template <typename T>
        struct is_complex<std::complex<T>> : std::true_type {};

        template <typename T>
        inline constexpr bool is_complex_v = is_complex<T>::value;
        
        template<typename Func>
        sycl::event execute_random(Engine& engine, Tensor& t, const char* name, SushiRuntime::Graph::OpID op_id, const std::vector<double>& params, Func&& task_func) 
        {
            const int64_t size = t.num_elements;
            void* ptr = t.storage ? t.storage->data_ptr : nullptr;
            
            std::vector<void*> reads = {};
            std::vector<void*> writes = {};
            if (ptr) writes.push_back(ptr);
            
            SushiRuntime::Graph::TaskMetadata meta;
            meta.name = name;
            meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
            meta.op_id = op_id;

            for (size_t i = 0; i < params.size(); ++i) {
                meta.set_param(i, params[i]);
            }

            const uint64_t seed = engine.get_seed();
            const uint64_t offset = engine.get_and_increment_rng_offset();
            
            meta.set_param(10, static_cast<double>(seed));
            meta.set_param(11, static_cast<double>(offset));

            engine.get_graph().add_task(meta, reads, writes,
                [dtype=t.dtype, size, seed, offset, task_func, pT = ptr, name](sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event 
                {
                    SB_LOG_INFO("RandomOps: {} ({} elements, seed: {}, offset: {})", name, size, seed, offset);
                    
                    switch (dtype) {
                        case Core::DataType::FLOAT32:
                            return task_func(float{}, q, seed, offset, size, static_cast<float*>(pT), deps);
                        case Core::DataType::FLOAT64:
                            return task_func(double{}, q, seed, offset, size, static_cast<double*>(pT), deps);
                        case Core::DataType::COMPLEX32:
                            return task_func(std::complex<float>{}, q, seed, offset, size, static_cast<std::complex<float>*>(pT), deps);
                        case Core::DataType::COMPLEX64:
                            return task_func(std::complex<double>{}, q, seed, offset, size, static_cast<std::complex<double>*>(pT), deps);
                        case Core::DataType::HALF:
                            SB_THROW_IF(true, "HALF precision is not natively supported by MKL RNG.");
                            return sycl::event();
                        default:
                            SB_THROW_IF(true, "Unsupported data type for RNG operation.");
                            return sycl::event();
                    }
                }
            );
            return sycl::event();
        }
    }
}
