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
         * @brief Records a unary in-place elementwise op: `t[i] = op(t[i])` for every element.
         *
         * The generic seam behind every `ElementwiseOps` unary entry point, and usable
         * directly by a downstream consumer that wants several elementwise steps collapsed
         * into a single kernel: compose them into one functor and call this once.
         *
         * @tparam Dom Which element types to instantiate the functor for. Defaults to
         *         Domain::REAL; pass Domain::REAL_AND_COMPLEX only when `op_func` is
         *         well-formed for `std::complex` (see safe_math.hpp).
         * @tparam Func The functor type, deduced.
         * @param engine The engine whose task graph the op is recorded on.
         * @param t The tensor read and overwritten in place.
         * @param name Human-readable op name, used for logging and failure messages.
         * @param op_id Stable operation identifier for the task graph.
         * @param op_func Callable `(auto x) -> auto`, invoked once per element on device.
         * @param params Optional scalar parameters recorded in the task metadata.
         * @return An empty sycl::event; the work is deferred until `Engine::execute()`.
         *
         * @warning This is a low-level generic seam, not a convenience wrapper. `op_func` is
         *          inlined into a SYCL device kernel, so it must be device-copyable and
         *          trivially copyable, must capture by value only, must not dereference host
         *          pointers or touch host-only state, must not allocate, and must not throw -
         *          none of which the compiler will always diagnose for you. Element type
         *          conversions are the functor's job: whatever it returns is assigned
         *          straight back into the tensor's storage.
         * @throws std::runtime_error if the tensor's dtype falls outside @p Dom, or if
         *         @p params exceeds the task metadata's inline capacity.
         */
        template<Domain Dom = Domain::REAL, typename Func>
        sycl::event execute_unary_inplace(Engine& engine, Tensor& t, const char* name, SushiRuntime::Graph::OpID op_id, Func&& op_func, const std::vector<float>& params = {})
        {
            // TODO: Optimize parallel_for with 1D/2D/3D nd_range, vectorization (float4), and local memory coalescing for faster elementwise capability.
            if constexpr (Dom == Domain::REAL)
            {
                SB_THROW_IF(Core::is_complex_dtype(t.dtype),
                            "Elementwise operation '{}' is not defined for complex tensors (dtype {}): it needs an "
                            "ordering or a real-only builtin that complex numbers do not provide.",
                            name, Core::to_string(t.dtype));
            }

            int64_t size = t.num_elements;
            void* ptr = t.storage ? t.storage->data_ptr : nullptr;
            std::vector<SushiRuntime::Core::ResourceRegion> rw = ptr ? std::vector<SushiRuntime::Core::ResourceRegion>{ptr} : std::vector<SushiRuntime::Core::ResourceRegion>{};

            SushiRuntime::Graph::TaskMetadata meta;
            meta.name = name;
            meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
            meta.op_id = op_id;
            set_task_params(meta, params, name);

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
                            case Core::DataType::COMPLEX32:
                                if constexpr (Dom == Domain::REAL_AND_COMPLEX)
                                {
                                    h.parallel_for(sycl::range<1>(size), [=, p = (std::complex<float>*)ptr](sycl::id<1> i) { p[i] = op_func(p[i]); });
                                }
                                else
                                {
                                    SB_THROW_IF(true, "Elementwise operation '{}' does not support dtype {}.", name, Core::to_string(dtype));
                                }
                                break;
                            case Core::DataType::COMPLEX64:
                                if constexpr (Dom == Domain::REAL_AND_COMPLEX)
                                {
                                    h.parallel_for(sycl::range<1>(size), [=, p = (std::complex<double>*)ptr](sycl::id<1> i) { p[i] = op_func(p[i]); });
                                }
                                else
                                {
                                    SB_THROW_IF(true, "Elementwise operation '{}' does not support dtype {}.", name, Core::to_string(dtype));
                                }
                                break;
                            default:
                                SB_THROW_IF(true, "Elementwise operation '{}' received an unrecognized dtype ({}).",
                                            name, static_cast<int>(dtype));
                                break;
                        }
                    });
                });
            return sycl::event();
        }

        /**
         * @brief Records a binary elementwise op: `C[i] = op(A[i], B[i])` for every element.
         *
         * The generic seam behind every `ElementwiseOps` binary entry point. A downstream
         * consumer fusing an expression tree into one functor and calling this once gets a
         * single SYCL kernel for the whole expression.
         *
         * @tparam Dom Which element types to instantiate the functor for. Defaults to
         *         Domain::REAL; pass Domain::REAL_AND_COMPLEX only when `op_func` is
         *         well-formed for `std::complex` (see safe_math.hpp).
         * @tparam Func The functor type, deduced.
         * @param engine The engine whose task graph the op is recorded on.
         * @param A The first input tensor.
         * @param B The second input tensor.
         * @param C The output tensor; may alias an input.
         * @param name Human-readable op name, used for logging and failure messages.
         * @param op_id Stable operation identifier for the task graph.
         * @param op_func Callable `(auto x, auto y) -> auto`, invoked once per element on device.
         * @param params Optional scalar parameters recorded in the task metadata.
         * @return An empty sycl::event; the work is deferred until `Engine::execute()`.
         *
         * @warning Same device-kernel constraints as @ref execute_unary_inplace: `op_func` is
         *          inlined into the kernel, so it must be device-copyable, capture by value,
         *          hold no host pointers, allocate nothing and throw nothing.
         * @throws std::runtime_error if the operands disagree on size or dtype, if the dtype
         *         falls outside @p Dom, or if @p params exceeds the metadata's inline capacity.
         */
        template<Domain Dom = Domain::REAL, typename Func>
        sycl::event execute_binary(Engine& engine, const Tensor& A, const Tensor& B, Tensor& C, const char* name, SushiRuntime::Graph::OpID op_id, Func&& op_func, const std::vector<float>& params = {})
        {
            // TODO: Optimize parallel_for with nd_range and vectorization (float4) for high-performance memory bandwidth utilization.
            SB_THROW_IF(A.num_elements != B.num_elements || A.num_elements != C.num_elements,
                    "Tensor sizes must match for elementwise operation.");
            SB_THROW_IF(A.dtype != B.dtype || A.dtype != C.dtype,
                    "Tensor data types must match for elementwise operation.");

            if constexpr (Dom == Domain::REAL)
            {
                SB_THROW_IF(Core::is_complex_dtype(A.dtype),
                            "Elementwise operation '{}' is not defined for complex tensors (dtype {}): it needs an "
                            "ordering or a real-only builtin that complex numbers do not provide.",
                            name, Core::to_string(A.dtype));
            }

            int64_t size = A.num_elements;
            std::vector<SushiRuntime::Core::ResourceRegion> reads = {A.storage->data_ptr, B.storage->data_ptr};
            std::vector<SushiRuntime::Core::ResourceRegion> writes = {C.storage->data_ptr};

            SushiRuntime::Graph::TaskMetadata meta;
            meta.name = name;
            meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
            meta.op_id = op_id;
            set_task_params(meta, params, name);

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
                            case Core::DataType::COMPLEX32:
                                if constexpr (Dom == Domain::REAL_AND_COMPLEX)
                                {
                                    h.parallel_for(sycl::range<1>(size), [=, pA = (std::complex<float>*)pA_raw, pB = (std::complex<float>*)pB_raw, pC = (std::complex<float>*)pC_raw](sycl::id<1> i) { pC[i] = op_func(pA[i], pB[i]); });
                                }
                                else
                                {
                                    SB_THROW_IF(true, "Elementwise operation '{}' does not support dtype {}.", name, Core::to_string(dtype));
                                }
                                break;
                            case Core::DataType::COMPLEX64:
                                if constexpr (Dom == Domain::REAL_AND_COMPLEX)
                                {
                                    h.parallel_for(sycl::range<1>(size), [=, pA = (std::complex<double>*)pA_raw, pB = (std::complex<double>*)pB_raw, pC = (std::complex<double>*)pC_raw](sycl::id<1> i) { pC[i] = op_func(pA[i], pB[i]); });
                                }
                                else
                                {
                                    SB_THROW_IF(true, "Elementwise operation '{}' does not support dtype {}.", name, Core::to_string(dtype));
                                }
                                break;
                            default:
                                SB_THROW_IF(true, "Elementwise operation '{}' received an unrecognized dtype ({}).",
                                            name, static_cast<int>(dtype));
                                break;
                        }
                    });
                });
            return sycl::event();
        }

        /**
         * @brief Addition, as a named functor for callers composing ops generically.
         * @param a Left operand.
         * @param b Right operand.
         * @return `a + b`.
         */
        template<typename T>
        inline T safe_ew_add(T a, T b) { return a + b; }

        /**
         * @brief Subtraction, as a named functor for callers composing ops generically.
         * @param a Left operand.
         * @param b Right operand.
         * @return `a - b`.
         */
        template<typename T>
        inline T safe_ew_sub(T a, T b) { return a - b; }

        /**
         * @brief Multiplication, as a named functor for callers composing ops generically.
         * @param a Left operand.
         * @param b Right operand.
         * @return `a * b`.
         */
        template<typename T>
        inline T safe_ew_mul(T a, T b) { return a * b; }

        /**
         * @brief Division, as a named functor for callers composing ops generically.
         * @param a Left operand.
         * @param b Right operand.
         * @return `a / b`.
         */
        template<typename T>
        inline T safe_ew_div(T a, T b) { return a / b; }

    } // namespace Dispatch
} // namespace SushiBLAS
