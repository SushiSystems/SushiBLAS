/**************************************************************************/
/* dispatch.hpp                                                           */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

#pragma once

#include <vector>
#include <complex>
#include <sycl/sycl.hpp>
#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/tensor.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/engine/dispatch.hpp>
#include <SushiBLAS/engine/math/safe_math.hpp>
#include <SushiRuntime/core/resource_region.hpp>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiBLAS
{
    namespace Dispatch
    {
        /**
         * @brief Records an activation's forward pass in place: `t[i] = op(t[i])`.
         *
         * The generic seam behind every `NonLinearOps` forward entry point. Unlike the
         * elementwise family this instantiates all five dtypes unconditionally, because
         * every activation in this library is expressed through the complex-tolerant
         * `safe_*` helpers and is therefore well-formed for complex element types too.
         *
         * @tparam Func The functor type, deduced.
         * @param engine The engine whose task graph the op is recorded on.
         * @param t The tensor read and overwritten in place.
         * @param name Human-readable op name, used for logging and failure messages.
         * @param op_id Stable operation identifier for the task graph.
         * @param op_func Callable `(auto x) -> auto`, invoked once per element on device.
         * @param params Optional scalar parameters recorded in the task metadata.
         * @return An empty sycl::event; the work is deferred until `Engine::execute()`.
         *
         * @warning This is a low-level generic seam. `op_func` is inlined into a SYCL device
         *          kernel: it must be device-copyable, capture by value only, hold no host
         *          pointers, allocate nothing and throw nothing. Write its body against
         *          <SushiBLAS/engine/math/safe_math.hpp> rather than the raw `sycl::` math
         *          builtins, or it will fail to compile for the complex dtypes.
         * @throws std::runtime_error if the tensor's dtype is unrecognized, or if @p params
         *         exceeds the task metadata's inline capacity.
         */
        template<typename Func>
        sycl::event execute_nonlinear_forward(Engine& engine, Tensor& t, const char* name, SushiRuntime::Graph::OpID op_id, Func&& op_func, const std::vector<float>& params = {})
        {
            int64_t size = t.num_elements;
            void* ptr = t.data_or_null();
            std::vector<SushiRuntime::Core::ResourceRegion> rw = ptr ? std::vector<SushiRuntime::Core::ResourceRegion>{t.region()} : std::vector<SushiRuntime::Core::ResourceRegion>{};

            SushiRuntime::Graph::TaskMetadata meta;
            meta.name = name;
            meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
            meta.op_id = op_id;
            set_task_params(meta, params, name);

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
                            default:
                                SB_THROW_IF(true, "Nonlinear operation '{}' received an unrecognized dtype ({}).",
                                            name, static_cast<int>(dtype));
                                break;
                        }
                    });
                });
            return sycl::event();
        }

        /**
         * @brief Records an activation's backward pass: `dx[i] = op(dy[i], x[i])`.
         *
         * The generic seam behind every `NonLinearOps` backward entry point. The second
         * operand is whatever the activation's derivative needs - the saved input for GELU
         * and ELU, the saved output for sigmoid and tanh - which is why it is named
         * generically here and documented per op at the call site.
         *
         * @tparam Func The functor type, deduced.
         * @param engine The engine whose task graph the op is recorded on.
         * @param dy The upstream gradient.
         * @param x The saved forward input or output the derivative is taken against.
         * @param dx The output gradient tensor.
         * @param name Human-readable op name, used for logging and failure messages.
         * @param op_id Stable operation identifier for the task graph.
         * @param op_func Callable `(auto dy, auto x) -> auto`, invoked once per element on device.
         * @param params Optional scalar parameters recorded in the task metadata.
         * @return An empty sycl::event; the work is deferred until `Engine::execute()`.
         *
         * @warning Same device-kernel constraints as @ref execute_nonlinear_forward.
         * @throws std::runtime_error if the operands disagree on size or dtype, if the dtype
         *         is unrecognized, or if @p params exceeds the metadata's inline capacity.
         */
        template<typename Func>
        sycl::event execute_nonlinear_backward(Engine& engine, const Tensor& dy, const Tensor& x, Tensor& dx, const char* name, SushiRuntime::Graph::OpID op_id, Func&& op_func, const std::vector<float>& params = {})
        {
            SB_THROW_IF(dy.num_elements != x.num_elements || dy.num_elements != dx.num_elements,
                       "Tensor sizes must match for backward operation.");
            SB_THROW_IF(dy.dtype != x.dtype || dy.dtype != dx.dtype,
                       "Tensor data types must match for backward operation.");

            int64_t size = x.num_elements;
            std::vector<SushiRuntime::Core::ResourceRegion> reads = {dy.region(), x.region()};
            std::vector<SushiRuntime::Core::ResourceRegion> writes = {dx.region()};

            SushiRuntime::Graph::TaskMetadata meta;
            meta.name = name;
            meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
            meta.op_id = op_id;
            set_task_params(meta, params, name);

            engine.get_graph().add_task(meta, reads, writes,
                [size, pDY_raw = dy.data(), pX_raw = x.data(), pDX_raw = dx.data(), dtype = x.dtype, op_func, name](sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
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
                            default:
                                SB_THROW_IF(true, "Nonlinear operation '{}' received an unrecognized dtype ({}).",
                                            name, static_cast<int>(dtype));
                                break;
                        }
                    });
                });
            return sycl::event();
        }

    } // namespace Dispatch
} // namespace SushiBLAS
