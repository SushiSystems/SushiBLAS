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
#include <cstddef>
#include <sycl/sycl.hpp>
#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/tensor.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/engine/dispatch.hpp>
#include <SushiBLAS/engine/launch_1d.hpp>
#include <SushiRuntime/core/resource_region.hpp>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiBLAS
{
    namespace Dispatch
    {
        /**
         * @brief Records a unary logic op: `result[i] = op(x[i])`.
         *
         * Logic results are carried in the operand's own dtype (1 for true, 0 for false)
         * rather than a dedicated boolean tensor type, so `result` must match `x`.
         *
         * @tparam Dom Which element types to instantiate the functor for. Defaults to
         *         Domain::REAL; pass Domain::REAL_AND_COMPLEX only when `op_func` is
         *         well-formed for `std::complex`.
         * @tparam Func The functor type, deduced.
         * @param engine The engine whose task graph the op is recorded on.
         * @param x The input tensor.
         * @param result The output tensor; must match `x` in size and dtype.
         * @param name Human-readable op name, used for logging and failure messages.
         * @param op_id Stable operation identifier for the task graph.
         * @param op_func Callable `(auto x) -> auto`, invoked once per element on device.
         * @param params Optional scalar parameters recorded in the task metadata.
         * @return An empty sycl::event; the work is deferred until `Engine::execute()`.
         *
         * @warning This is a low-level generic seam. `op_func` is inlined into a SYCL device
         *          kernel: it must be device-copyable, capture by value only, hold no host
         *          pointers, allocate nothing and throw nothing.
         * @throws std::runtime_error if the operands disagree on size or dtype, if the dtype
         *         falls outside @p Dom, or if @p params exceeds the metadata's inline capacity.
         */
        template<Domain Dom = Domain::REAL, typename Func>
        sycl::event execute_logic_unary(Engine& engine, const Tensor& x, Tensor& result, const char* name, SushiRuntime::Graph::OpID op_id, Func&& op_func, const std::vector<float>& params = {})
        {
            SB_THROW_IF(x.num_elements != result.num_elements, "Tensor sizes must match for logic operation.");
            SB_THROW_IF(x.dtype != result.dtype, "Data types must match for logic operation.");

            require_domain<Dom>(x.dtype, "Logic operation", name);

            int64_t size = x.num_elements;
            std::vector<SushiRuntime::Core::ResourceRegion> reads = {x.region()};
            std::vector<SushiRuntime::Core::ResourceRegion> writes = {result.region()};
            const bool vec_ok = Launch::vector_aligned(x.data(), result.data());

            SushiRuntime::Graph::TaskMetadata meta;
            meta.name = name;
            meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
            meta.op_id = op_id;
            set_task_params(meta, params, name);

            engine.get_graph().add_task(meta, reads, writes,
                [size, vec_ok, px_raw = x.data(), pr_raw = result.data(), dtype = x.dtype, op_func, name](sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
                {
                    SB_LOG_INFO("Logic {}: {} elements", name, size);
                    return q.submit([&](sycl::handler& h)
                    {
                        h.depends_on(deps);
                        const std::size_t n = static_cast<std::size_t>(size);
                        Dispatch::with_element_type<Dom>(dtype, "Logic operation", name,
                            [&](auto tag)
                        {
                            using T = typename decltype(tag)::type;

                            Launch::map(h, q, n, vec_ok, op_func, (T*)pr_raw, (const T*)px_raw);
                        });
                    });
                });
            return sycl::event();
        }

        /**
         * @brief Records a binary logic op: `result[i] = op(A[i], B[i])`.
         *
         * @tparam Dom Which element types to instantiate the functor for. Defaults to
         *         Domain::REAL; the ordering comparisons keep that default deliberately,
         *         since `<` and `>` have no meaning on complex numbers.
         * @tparam Func The functor type, deduced.
         * @param engine The engine whose task graph the op is recorded on.
         * @param A The first input tensor.
         * @param B The second input tensor.
         * @param result The output tensor; must match the inputs in size and dtype.
         * @param name Human-readable op name, used for logging and failure messages.
         * @param op_id Stable operation identifier for the task graph.
         * @param op_func Callable `(auto x, auto y) -> auto`, invoked once per element on device.
         * @param params Optional scalar parameters recorded in the task metadata.
         * @return An empty sycl::event; the work is deferred until `Engine::execute()`.
         *
         * @warning Same device-kernel constraints as @ref execute_logic_unary.
         * @throws std::runtime_error if the operands disagree on size or dtype, if the dtype
         *         falls outside @p Dom, or if @p params exceeds the metadata's inline capacity.
         */
        template<Domain Dom = Domain::REAL, typename Func>
        sycl::event execute_logic_binary(Engine& engine, const Tensor& A, const Tensor& B, Tensor& result, const char* name, SushiRuntime::Graph::OpID op_id, Func&& op_func, const std::vector<float>& params = {})
        {
            SB_THROW_IF(A.num_elements != B.num_elements || A.num_elements != result.num_elements, "Tensor sizes must match for logic operation.");
            SB_THROW_IF(A.dtype != B.dtype || A.dtype != result.dtype, "Data types must match for logic operation.");

            require_domain<Dom>(A.dtype, "Logic operation", name);

            int64_t size = A.num_elements;
            std::vector<SushiRuntime::Core::ResourceRegion> reads = {A.region(), B.region()};
            std::vector<SushiRuntime::Core::ResourceRegion> writes = {result.region()};
            const bool vec_ok = Launch::vector_aligned(A.data(), B.data(), result.data());

            SushiRuntime::Graph::TaskMetadata meta;
            meta.name = name;
            meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
            meta.op_id = op_id;
            set_task_params(meta, params, name);

            engine.get_graph().add_task(meta, reads, writes,
                [size, vec_ok, pA_raw = A.data(), pB_raw = B.data(), pR_raw = result.data(), dtype = A.dtype, op_func, name](sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
                {
                    SB_LOG_INFO("Logic {}: {} elements", name, size);
                    return q.submit([&](sycl::handler& h)
                    {
                        h.depends_on(deps);
                        const std::size_t n = static_cast<std::size_t>(size);
                        Dispatch::with_element_type<Dom>(dtype, "Logic operation", name,
                            [&](auto tag)
                        {
                            using T = typename decltype(tag)::type;

                            Launch::map(h, q, n, vec_ok, op_func, (T*)pR_raw, (const T*)pA_raw, (const T*)pB_raw);
                        });
                    });
                });
            return sycl::event();
        }

        /**
         * @brief Records a ternary logic op: `result[i] = op(cond[i], A[i], B[i])`.
         *
         * @tparam Dom Which element types to instantiate the functor for. Defaults to
         *         Domain::REAL; pass Domain::REAL_AND_COMPLEX only when `op_func` is
         *         well-formed for `std::complex`.
         * @tparam Func The functor type, deduced.
         * @param engine The engine whose task graph the op is recorded on.
         * @param cond The selector tensor; nonzero selects `A`.
         * @param A The value taken where `cond` is nonzero.
         * @param B The value taken where `cond` is zero.
         * @param result The output tensor; must match the inputs in size and dtype.
         * @param name Human-readable op name, used for logging and failure messages.
         * @param op_id Stable operation identifier for the task graph.
         * @param op_func Callable `(auto c, auto a, auto b) -> auto`, invoked once per element.
         * @param params Optional scalar parameters recorded in the task metadata.
         * @return An empty sycl::event; the work is deferred until `Engine::execute()`.
         *
         * @warning Same device-kernel constraints as @ref execute_logic_unary.
         * @throws std::runtime_error if the operands disagree on size or dtype, if the dtype
         *         falls outside @p Dom, or if @p params exceeds the metadata's inline capacity.
         */
        template<Domain Dom = Domain::REAL, typename Func>
        sycl::event execute_logic_ternary(Engine& engine, const Tensor& cond, const Tensor& A, const Tensor& B, Tensor& result, const char* name, SushiRuntime::Graph::OpID op_id, Func&& op_func, const std::vector<float>& params = {})
        {
            SB_THROW_IF(cond.num_elements != A.num_elements || cond.num_elements != B.num_elements || cond.num_elements != result.num_elements, "Tensor sizes must match for logic operation.");
            SB_THROW_IF(cond.dtype != A.dtype || cond.dtype != B.dtype || cond.dtype != result.dtype, "Data types must match for logic operation.");

            require_domain<Dom>(cond.dtype, "Logic operation", name);

            int64_t size = cond.num_elements;
            std::vector<SushiRuntime::Core::ResourceRegion> reads = {cond.region(), A.region(), B.region()};
            std::vector<SushiRuntime::Core::ResourceRegion> writes = {result.region()};
            const bool vec_ok = Launch::vector_aligned(cond.data(), A.data(), B.data(), result.data());

            SushiRuntime::Graph::TaskMetadata meta;
            meta.name = name;
            meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
            meta.op_id = op_id;
            set_task_params(meta, params, name);

            engine.get_graph().add_task(meta, reads, writes,
                [size, vec_ok, pC_raw = cond.data(), pA_raw = A.data(), pB_raw = B.data(), pR_raw = result.data(), dtype = cond.dtype, op_func, name](sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
                {
                    SB_LOG_INFO("Logic {}: {} elements", name, size);
                    return q.submit([&](sycl::handler& h)
                    {
                        h.depends_on(deps);
                        const std::size_t n = static_cast<std::size_t>(size);
                        Dispatch::with_element_type<Dom>(dtype, "Logic operation", name,
                            [&](auto tag)
                        {
                            using T = typename decltype(tag)::type;

                            Launch::map(h, q, n, vec_ok, op_func, (T*)pR_raw, (const T*)pC_raw, (const T*)pA_raw, (const T*)pB_raw);
                        });
                    });
                });
            return sycl::event();
        }

        /**
         * @brief Truth test for a real scalar.
         * @param x The value to test.
         * @return True if `x` is not zero.
         */
        template<typename T>
        inline bool logic_not_zero(T x) { return x != T(0); }

        /**
         * @brief Truth test for a complex value.
         * @param x The value to test.
         * @return True if either component of `x` is nonzero.
         */
        template<typename T>
        inline bool logic_not_zero(const std::complex<T>& x) { return x != std::complex<T>(0, 0); }

        /**
         * @brief Equality of two real scalars.
         * @param x Left operand.
         * @param y Right operand.
         * @return True if `x == y`.
         */
        template<typename T>
        inline bool logic_equal(T x, T y) { return x == y; }

        /**
         * @brief Equality of two complex values, component-wise.
         * @param x Left operand.
         * @param y Right operand.
         * @return True if both components match.
         */
        template<typename T>
        inline bool logic_equal(const std::complex<T>& x, const std::complex<T>& y) { return x == y; }

        // No complex overloads below this line, on purpose. `<`, `<=`, `>` and `>=` have no
        // meaning on complex numbers; these helpers used to supply a lexicographic stand-in,
        // which made `less(a, b)` on a COMPLEX32 tensor return a confident, well-formed,
        // mathematically meaningless answer. The ordering ops now declare Domain::REAL, so a
        // complex operand throws at record time instead of being quietly re-interpreted.

        /**
         * @brief Less-than comparison of two real scalars.
         * @param x Left operand.
         * @param y Right operand.
         * @return True if `x < y`.
         */
        template<typename T>
        inline bool logic_less(T x, T y) { return x < y; }

        /**
         * @brief Greater-than comparison of two real scalars.
         * @param x Left operand.
         * @param y Right operand.
         * @return True if `x > y`.
         */
        template<typename T>
        inline bool logic_greater(T x, T y) { return x > y; }

        /**
         * @brief Less-than-or-equal comparison of two real scalars.
         * @param x Left operand.
         * @param y Right operand.
         * @return True if `x <= y`.
         */
        template<typename T>
        inline bool logic_less_equal(T x, T y) { return x <= y; }

        /**
         * @brief Greater-than-or-equal comparison of two real scalars.
         * @param x Left operand.
         * @param y Right operand.
         * @return True if `x >= y`.
         */
        template<typename T>
        inline bool logic_greater_equal(T x, T y) { return x >= y; }

    } // namespace Dispatch
} // namespace SushiBLAS
