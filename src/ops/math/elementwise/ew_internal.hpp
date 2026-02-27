/**************************************************************************/
/* ew_internal.hpp                                                        */
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
#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/tensor.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiBLAS
{
    using namespace SushiRuntime::Graph::Literals;

    namespace Internal 
    {
        template<typename T> struct is_complex : std::false_type {};
        template<typename T> struct is_complex<std::complex<T>> : std::true_type {};
        template<typename T> inline constexpr bool is_complex_v = is_complex<T>::value;
        

        /**
        * @brief Helper for unary in-place elementwise operations.
        */
        template<typename Func>
        sycl::event execute_unary_inplace(Engine& engine, Tensor& t, const char* name, SushiRuntime::Graph::OpID op_id, Func&& op_func, const std::vector<float>& params = {}) 
        {
            int64_t size = t.num_elements;
            void* ptr = t.storage ? t.storage->data_ptr : nullptr;
            std::vector<void*> rw = ptr ? std::vector<void*>{ptr} : std::vector<void*>{};

            SushiRuntime::Graph::TaskMetadata meta;
            meta.name = name;
            meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
            meta.op_id = op_id;
            for (size_t i = 0; i < params.size(); ++i) meta.set_param(i, params[i]);

            engine.get_graph().add_task(meta, rw, rw,
                [size, ptr, dtype = t.dtype, op_func, name](sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event 
                {
                    SB_LOG_INFO("Elementwise {} (In-place): {} elements", name, size);
                    return q.submit([&](sycl::handler& h) 
                    {
                        h.depends_on(deps);
                        switch (dtype) 
                        {
                            case Core::DataType::HALF:
                                h.parallel_for(sycl::range<1>(size), [=, p = (sycl::half*)ptr](sycl::id<1> i) { p[i] = op_func(p[i]); });
                                break;
                            case Core::DataType::FLOAT32:
                                h.parallel_for(sycl::range<1>(size), [=, p = (float*)ptr](sycl::id<1> i) { p[i] = op_func(p[i]); });
                                break;
                            case Core::DataType::FLOAT64:
                                h.parallel_for(sycl::range<1>(size), [=, p = (double*)ptr](sycl::id<1> i) { p[i] = op_func(p[i]); });
                                break;
                            default: break;
                        }
                    });
                });
            return sycl::event();
        }

        /**
        * @brief Helper for binary elementwise operations (C = op(A, B)).
        */
        template<typename Func>
        sycl::event execute_binary(Engine& engine, const Tensor& A, const Tensor& B, Tensor& C, const char* name, SushiRuntime::Graph::OpID op_id, Func&& op_func, const std::vector<float>& params = {}) 
        {
            SB_THROW_IF(A.num_elements != B.num_elements || A.num_elements != C.num_elements, 
                    "Tensor sizes must match for elementwise operation.");
            SB_THROW_IF(A.dtype != B.dtype || A.dtype != C.dtype, 
                    "Tensor data types must match for elementwise operation.");

            int64_t size = A.num_elements;
            std::vector<void*> reads = {A.storage->data_ptr, B.storage->data_ptr};
            std::vector<void*> writes = {C.storage->data_ptr};

            SushiRuntime::Graph::TaskMetadata meta;
            meta.name = name;
            meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
            meta.op_id = op_id;
            for (size_t i = 0; i < params.size(); ++i) meta.set_param(i, params[i]);

            engine.get_graph().add_task(meta, reads, writes,
                [size, pA_raw = A.storage->data_ptr, pB_raw = B.storage->data_ptr, pC_raw = C.storage->data_ptr, dtype = A.dtype, op_func, name](sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event 
                {
                    SB_LOG_INFO("Elementwise {}: {} elements", name, size);
                    return q.submit([&](sycl::handler& h) 
                    {
                        h.depends_on(deps);
                        switch (dtype) 
                        {
                            case Core::DataType::HALF:
                                h.parallel_for(sycl::range<1>(size), [=, pA = (sycl::half*)pA_raw, pB = (sycl::half*)pB_raw, pC = (sycl::half*)pC_raw](sycl::id<1> i) { pC[i] = op_func(pA[i], pB[i]); });
                                break;
                            case Core::DataType::FLOAT32:
                                h.parallel_for(sycl::range<1>(size), [=, pA = (float*)pA_raw, pB = (float*)pB_raw, pC = (float*)pC_raw](sycl::id<1> i) { pC[i] = op_func(pA[i], pB[i]); });
                                break;
                            case Core::DataType::FLOAT64:
                                h.parallel_for(sycl::range<1>(size), [=, pA = (double*)pA_raw, pB = (double*)pB_raw, pC = (double*)pC_raw](sycl::id<1> i) { pC[i] = op_func(pA[i], pB[i]); });
                                break;
                            default: break;
                        }
                    });
                });
            return sycl::event();
        }

        template<typename T>
        inline T safe_ew_add(T a, T b) { return a + b; }
        template<typename T>
        inline T safe_ew_sub(T a, T b) { return a - b; }
        template<typename T>
        inline T safe_ew_mul(T a, T b) { return a * b; }
        template<typename T>
        inline T safe_ew_div(T a, T b) { return a / b; }

    } // namespace Internal
} // namespace SushiBLAS
