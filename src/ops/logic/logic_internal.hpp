/**************************************************************************/
/* logic_internal.hpp                                                     */
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
        /**
         * @brief Helper for unary logic operations.
         */
        template<typename Func>
        sycl::event execute_logic_unary(Engine& engine, const Tensor& x, Tensor& result, const char* name, SushiRuntime::Graph::OpID op_id, Func&& op_func, const std::vector<float>& params = {}) 
        {
            SB_THROW_IF(x.num_elements != result.num_elements, "Tensor sizes must match for logic operation.");
            SB_THROW_IF(x.dtype != result.dtype, "Data types must match for logic operation.");

            int64_t size = x.num_elements;
            std::vector<void*> reads = {x.storage->data_ptr};
            std::vector<void*> writes = {result.storage->data_ptr};

            SushiRuntime::Graph::TaskMetadata meta;
            meta.name = name;
            meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
            meta.op_id = op_id;
            for (size_t i = 0; i < params.size(); ++i) meta.set_param(i, params[i]);

            engine.get_graph().add_task(meta, reads, writes,
                [size, px_raw = x.storage->data_ptr, pr_raw = result.storage->data_ptr, dtype = x.dtype, op_func, name](sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event 
                {
                    SB_LOG_INFO("Logic {}: {} elements", name, size);
                    return q.submit([&](sycl::handler& h) 
                    {
                        h.depends_on(deps);
                        switch (dtype) 
                        {
                            case Core::DataType::HALF:
                                h.parallel_for(sycl::range<1>(size), [=, px = (sycl::half*)px_raw, pr = (sycl::half*)pr_raw](sycl::id<1> i) { pr[i] = op_func(px[i]); });
                                break;
                            case Core::DataType::FLOAT32:
                                h.parallel_for(sycl::range<1>(size), [=, px = (float*)px_raw, pr = (float*)pr_raw](sycl::id<1> i) { pr[i] = op_func(px[i]); });
                                break;
                            case Core::DataType::FLOAT64:
                                h.parallel_for(sycl::range<1>(size), [=, px = (double*)px_raw, pr = (double*)pr_raw](sycl::id<1> i) { pr[i] = op_func(px[i]); });
                                break;
                            case Core::DataType::COMPLEX32:
                                h.parallel_for(sycl::range<1>(size), [=, px = (std::complex<float>*)px_raw, pr = (std::complex<float>*)pr_raw](sycl::id<1> i) { pr[i] = op_func(px[i]); });
                                break;
                            case Core::DataType::COMPLEX64:
                                h.parallel_for(sycl::range<1>(size), [=, px = (std::complex<double>*)px_raw, pr = (std::complex<double>*)pr_raw](sycl::id<1> i) { pr[i] = op_func(px[i]); });
                                break;
                            default: break;
                        }
                    });
                });
            return sycl::event();
        }

        /**
         * @brief Helper for binary logic operations.
         */
        template<typename Func>
        sycl::event execute_logic_binary(Engine& engine, const Tensor& A, const Tensor& B, Tensor& result, const char* name, SushiRuntime::Graph::OpID op_id, Func&& op_func, const std::vector<float>& params = {}) 
        {
            SB_THROW_IF(A.num_elements != B.num_elements || A.num_elements != result.num_elements, "Tensor sizes must match for logic operation.");
            SB_THROW_IF(A.dtype != B.dtype || A.dtype != result.dtype, "Data types must match for logic operation.");

            int64_t size = A.num_elements;
            std::vector<void*> reads = {A.storage->data_ptr, B.storage->data_ptr};
            std::vector<void*> writes = {result.storage->data_ptr};

            SushiRuntime::Graph::TaskMetadata meta;
            meta.name = name;
            meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
            meta.op_id = op_id;
            for (size_t i = 0; i < params.size(); ++i) meta.set_param(i, params[i]);

            engine.get_graph().add_task(meta, reads, writes,
                [size, pA_raw = A.storage->data_ptr, pB_raw = B.storage->data_ptr, pR_raw = result.storage->data_ptr, dtype = A.dtype, op_func, name](sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event 
                {
                    SB_LOG_INFO("Logic {}: {} elements", name, size);
                    return q.submit([&](sycl::handler& h) 
                    {
                        h.depends_on(deps);
                        switch (dtype) 
                        {
                            case Core::DataType::HALF:
                                h.parallel_for(sycl::range<1>(size), [=, pA = (sycl::half*)pA_raw, pB = (sycl::half*)pB_raw, pR = (sycl::half*)pR_raw](sycl::id<1> i) { pR[i] = op_func(pA[i], pB[i]); });
                                break;
                            case Core::DataType::FLOAT32:
                                h.parallel_for(sycl::range<1>(size), [=, pA = (float*)pA_raw, pB = (float*)pB_raw, pR = (float*)pR_raw](sycl::id<1> i) { pR[i] = op_func(pA[i], pB[i]); });
                                break;
                            case Core::DataType::FLOAT64:
                                h.parallel_for(sycl::range<1>(size), [=, pA = (double*)pA_raw, pB = (double*)pB_raw, pR = (double*)pR_raw](sycl::id<1> i) { pR[i] = op_func(pA[i], pB[i]); });
                                break;
                            case Core::DataType::COMPLEX32:
                                h.parallel_for(sycl::range<1>(size), [=, pA = (std::complex<float>*)pA_raw, pB = (std::complex<float>*)pB_raw, pR = (std::complex<float>*)pR_raw](sycl::id<1> i) { pR[i] = op_func(pA[i], pB[i]); });
                                break;
                            case Core::DataType::COMPLEX64:
                                h.parallel_for(sycl::range<1>(size), [=, pA = (std::complex<double>*)pA_raw, pB = (std::complex<double>*)pB_raw, pR = (std::complex<double>*)pR_raw](sycl::id<1> i) { pR[i] = op_func(pA[i], pB[i]); });
                                break;
                            default: break;
                        }
                    });
                });
            return sycl::event();
        }

        template<typename T>
        inline bool logic_not_zero(T x) { return x != T(0); }
        
        template<typename T>
        inline bool logic_not_zero(const std::complex<T>& x) { return x != std::complex<T>(0, 0); }

        template<typename T>
        inline bool logic_equal(T x, T y) { return x == y; }
        
        template<typename T>
        inline bool logic_equal(const std::complex<T>& x, const std::complex<T>& y) { return x == y; }

        template<typename T>
        inline bool logic_less(T x, T y) { return x < y; }
        
        template<typename T>
        inline bool logic_less(const std::complex<T>& x, const std::complex<T>& y) { return x.real() < y.real() || (x.real() == y.real() && x.imag() < y.imag()); }

        template<typename T>
        inline bool logic_greater(T x, T y) { return x > y; }
        
        template<typename T>
        inline bool logic_greater(const std::complex<T>& x, const std::complex<T>& y) { return x.real() > y.real() || (x.real() == y.real() && x.imag() > y.imag()); }
        
        template<typename T>
        inline bool logic_less_equal(T x, T y) { return x <= y; }
        
        template<typename T>
        inline bool logic_less_equal(const std::complex<T>& x, const std::complex<T>& y) { return logic_less(x, y) || logic_equal(x, y); }

        template<typename T>
        inline bool logic_greater_equal(T x, T y) { return x >= y; }
        
        template<typename T>
        inline bool logic_greater_equal(const std::complex<T>& x, const std::complex<T>& y) { return logic_greater(x, y) || logic_equal(x, y); }

        template<typename Func>
        sycl::event execute_logic_ternary(Engine& engine, const Tensor& cond, const Tensor& A, const Tensor& B, Tensor& result, const char* name, SushiRuntime::Graph::OpID op_id, Func&& op_func, const std::vector<float>& params = {}) 
        {
            SB_THROW_IF(cond.num_elements != A.num_elements || cond.num_elements != B.num_elements || cond.num_elements != result.num_elements, "Tensor sizes must match for logic operation.");
            SB_THROW_IF(cond.dtype != A.dtype || cond.dtype != B.dtype || cond.dtype != result.dtype, "Data types must match for logic operation.");

            int64_t size = cond.num_elements;
            std::vector<void*> reads = {cond.storage->data_ptr, A.storage->data_ptr, B.storage->data_ptr};
            std::vector<void*> writes = {result.storage->data_ptr};

            SushiRuntime::Graph::TaskMetadata meta;
            meta.name = name;
            meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
            meta.op_id = op_id;
            for (size_t i = 0; i < params.size(); ++i) meta.set_param(i, params[i]);

            engine.get_graph().add_task(meta, reads, writes,
                [size, pC_raw = cond.storage->data_ptr, pA_raw = A.storage->data_ptr, pB_raw = B.storage->data_ptr, pR_raw = result.storage->data_ptr, dtype = cond.dtype, op_func, name](sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event 
                {
                    SB_LOG_INFO("Logic {}: {} elements", name, size);
                    return q.submit([&](sycl::handler& h) 
                    {
                        h.depends_on(deps);
                        switch (dtype) 
                        {
                            case Core::DataType::HALF:
                                h.parallel_for(sycl::range<1>(size), [=, pC = (sycl::half*)pC_raw, pA = (sycl::half*)pA_raw, pB = (sycl::half*)pB_raw, pR = (sycl::half*)pR_raw](sycl::id<1> i) { pR[i] = op_func(pC[i], pA[i], pB[i]); });
                                break;
                            case Core::DataType::FLOAT32:
                                h.parallel_for(sycl::range<1>(size), [=, pC = (float*)pC_raw, pA = (float*)pA_raw, pB = (float*)pB_raw, pR = (float*)pR_raw](sycl::id<1> i) { pR[i] = op_func(pC[i], pA[i], pB[i]); });
                                break;
                            case Core::DataType::FLOAT64:
                                h.parallel_for(sycl::range<1>(size), [=, pC = (double*)pC_raw, pA = (double*)pA_raw, pB = (double*)pB_raw, pR = (double*)pR_raw](sycl::id<1> i) { pR[i] = op_func(pC[i], pA[i], pB[i]); });
                                break;
                            case Core::DataType::COMPLEX32:
                                h.parallel_for(sycl::range<1>(size), [=, pC = (std::complex<float>*)pC_raw, pA = (std::complex<float>*)pA_raw, pB = (std::complex<float>*)pB_raw, pR = (std::complex<float>*)pR_raw](sycl::id<1> i) { pR[i] = op_func(pC[i], pA[i], pB[i]); });
                                break;
                            case Core::DataType::COMPLEX64:
                                h.parallel_for(sycl::range<1>(size), [=, pC = (std::complex<double>*)pC_raw, pA = (std::complex<double>*)pA_raw, pB = (std::complex<double>*)pB_raw, pR = (std::complex<double>*)pR_raw](sycl::id<1> i) { pR[i] = op_func(pC[i], pA[i], pB[i]); });
                                break;
                            default: break;
                        }
                    });
                });
            return sycl::event();
        }

    } // namespace Internal
} // namespace SushiBLAS
