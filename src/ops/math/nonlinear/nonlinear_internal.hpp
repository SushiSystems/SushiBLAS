/**************************************************************************/
/* nonlinear_internal.hpp                                                 */
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
         * @brief Helper for nonlinear forward operations.
         */
        template<typename Func>
        sycl::event execute_nonlinear_forward(Engine& engine, Tensor& t, const char* name, SushiRuntime::Graph::OpID op_id, Func&& op_func, const std::vector<float>& params = {}) 
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
                    SB_LOG_INFO("{} Forward: {} elements", name, size);
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
                            case Core::DataType::COMPLEX32:
                                h.parallel_for(sycl::range<1>(size), [=, p = (std::complex<float>*)ptr](sycl::id<1> i) { p[i] = op_func(p[i]); });
                                break;
                            case Core::DataType::COMPLEX64:
                                h.parallel_for(sycl::range<1>(size), [=, p = (std::complex<double>*)ptr](sycl::id<1> i) { p[i] = op_func(p[i]); });
                                break;
                            default: break;
                        }
                    });
                });
            return sycl::event();
        }

        /**
         * @brief Helper for nonlinear backward operations.
         */
        template<typename Func>
        sycl::event execute_nonlinear_backward(Engine& engine, const Tensor& dy, const Tensor& x, Tensor& dx, const char* name, SushiRuntime::Graph::OpID op_id, Func&& op_func, const std::vector<float>& params = {}) 
        {
            SB_THROW_IF(dy.num_elements != x.num_elements || dy.num_elements != dx.num_elements, 
                       "Tensor sizes must match for backward operation.");
            SB_THROW_IF(dy.dtype != x.dtype || dy.dtype != dx.dtype, 
                       "Tensor data types must match for backward operation.");

            int64_t size = x.num_elements;
            std::vector<void*> reads = {dy.storage->data_ptr, x.storage->data_ptr};
            std::vector<void*> writes = {dx.storage->data_ptr};

            SushiRuntime::Graph::TaskMetadata meta;
            meta.name = name;
            meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
            meta.op_id = op_id;
            for (size_t i = 0; i < params.size(); ++i) meta.set_param(i, params[i]);

            engine.get_graph().add_task(meta, reads, writes,
                [size, pDY_raw = dy.storage->data_ptr, pX_raw = x.storage->data_ptr, pDX_raw = dx.storage->data_ptr, dtype = x.dtype, op_func, name](sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event 
                {
                    SB_LOG_INFO("{} Backward: {} elements", name, size);
                    return q.submit([&](sycl::handler& h) 
                    {
                        h.depends_on(deps);
                        switch (dtype) 
                        {
                            case Core::DataType::HALF:
                                h.parallel_for(sycl::range<1>(size), [=, pdy = (sycl::half*)pDY_raw, px = (sycl::half*)pX_raw, pdx = (sycl::half*)pDX_raw](sycl::id<1> i) { pdx[i] = op_func(pdy[i], px[i]); });
                                break;
                            case Core::DataType::FLOAT32:
                                h.parallel_for(sycl::range<1>(size), [=, pdy = (float*)pDY_raw, px = (float*)pX_raw, pdx = (float*)pDX_raw](sycl::id<1> i) { pdx[i] = op_func(pdy[i], px[i]); });
                                break;
                            case Core::DataType::FLOAT64:
                                h.parallel_for(sycl::range<1>(size), [=, pdy = (double*)pDY_raw, px = (double*)pX_raw, pdx = (double*)pDX_raw](sycl::id<1> i) { pdx[i] = op_func(pdy[i], px[i]); });
                                break;
                            case Core::DataType::COMPLEX32:
                                h.parallel_for(sycl::range<1>(size), [=, pdy = (std::complex<float>*)pDY_raw, px = (std::complex<float>*)pX_raw, pdx = (std::complex<float>*)pDX_raw](sycl::id<1> i) { pdx[i] = op_func(pdy[i], px[i]); });
                                break;
                            case Core::DataType::COMPLEX64:
                                h.parallel_for(sycl::range<1>(size), [=, pdy = (std::complex<double>*)pDY_raw, px = (std::complex<double>*)pX_raw, pdx = (std::complex<double>*)pDX_raw](sycl::id<1> i) { pdx[i] = op_func(pdy[i], px[i]); });
                                break;
                            default: break;
                        }
                    });
                });
            return sycl::event();
        }

        // Helpers to apply math ops to both real and complex gracefully
        template<typename T>
        inline T real_part(T x) { return x; }
        
        template<typename T>
        inline std::complex<T> real_part(const std::complex<T>& x) { return std::complex<T>(x.real(), 0); }
        
        template<typename T>
        inline T make_complex_safe(T val, T) { return val; }

        template<typename T, typename U>
        inline std::complex<T> make_complex_safe(U val, const std::complex<T>&) { return std::complex<T>(val, 0); }
        
        template<typename T>
        inline bool greater_than_zero(T x) { return x > T(0); }

        template<typename T>
        inline bool greater_than_zero(const std::complex<T>& x) { return x.real() > T(0); }
        
        template<typename T>
        inline bool less_than_zero(T x) { return x < T(0); }

        template<typename T>
        inline bool less_than_zero(const std::complex<T>& x) { return x.real() < T(0); }

        template<typename T>
        inline T safe_exp(T x) { return sycl::exp(x); }
        template<typename T>
        inline std::complex<T> safe_exp(const std::complex<T>& x) { return std::exp(x); }
        
        template<typename T>
        inline T safe_tanh(T x) { return sycl::tanh(x); }
        template<typename T>
        inline std::complex<T> safe_tanh(const std::complex<T>& x) { return std::tanh(x); }
        
        template<typename T>
        inline T safe_erf(T x) { return sycl::erf(x); }
        template<typename T>
        inline std::complex<T> safe_erf(const std::complex<T>& x) { return std::complex<T>(std::erf(x.real()), 0); }

        template<typename T>
        inline T safe_log(T x) { return sycl::log(x); }
        template<typename T>
        inline std::complex<T> safe_log(const std::complex<T>& x) { return std::log(x); }

    } // namespace Internal
} // namespace SushiBLAS
