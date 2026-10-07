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
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
/**************************************************************************/

/**
 * @file dispatch.hpp
 * @brief Declares the logic executors of the dispatch seam and the comparison helpers they share.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <complex>
#include <cstddef>
#include <sycl/sycl.hpp>
#include <SushiBLAS/tensor.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/engine/dispatch.hpp>
#include <SushiBLAS/support/launch_1d.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/task_handle.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiRuntime/core/resource_region.hpp>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiBLAS
{
    namespace Dispatch
    {
        /**
         * @brief Records a unary logic op `result[i] = op(x[i])`, or nothing when result is empty.
         * @param result Must match x in size and dtype.
         * @param op_func Callable `(auto x) -> auto`: device-copyable, by-value captures, no allocation, no throw.
         * @throws std::invalid_argument on a missing storage, a non-contiguous operand, a shape or layout mismatch, or under FORCE_REFERENCE.
         * @throws std::runtime_error on a dtype mismatch, a dtype outside Dom, or too many params.
         * @see include/SushiBLAS/engine/logic/README.md
         */
        template<Domain Dom = Domain::REAL, typename Func, typename Param = double>
        TaskHandle execute_logic_unary(TaskRecorder& recorder, const Tensor& x, Tensor& result,
                                       const char* name, SushiRuntime::Graph::OpID op_id,
                                       Func&& op_func, const std::vector<Param>& params = {})
        {
            require_storage(x, name, "x");
            require_storage(result, name, "result");
            require_contiguous(x, name, "x");
            require_contiguous(result, name, "result");
            require_same_shape(x, result, name, "x", "result");
            require_same_layout(x, result, name, "x", "result");
            SB_THROW_IF(x.dtype != result.dtype, "Data types must match for logic operation.");

            require_domain<Dom>(x.dtype, "Logic operation", name);

            if (result.num_elements == 0)
                return TaskHandle();

            int64_t size = x.num_elements;
            OperandSet operands;
            operands.reads(x).writes(result);
            const bool vec_ok = Launch::vector_aligned(x.data(), result.data());

            SushiRuntime::Graph::TaskMetadata meta;
            meta.name = name;
            meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
            meta.op_id = op_id;
            set_task_params(meta, params, name);

            return recorder.record(meta, operands, "Logic operation",
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
        }

        /**
         * @brief Records a binary logic op `result[i] = op(A[i], B[i])`, or nothing when result is empty.
         * @tparam Dom Defaults to Domain::REAL, because `<` and `>` have no meaning on complex numbers.
         * @param result Must match the inputs in size and dtype.
         * @param op_func Callable `(auto x, auto y) -> auto` under execute_logic_unary's constraints.
         * @throws std::invalid_argument on a missing storage, a non-contiguous operand, a shape or layout mismatch, or under FORCE_REFERENCE.
         * @throws std::runtime_error on a dtype mismatch, a dtype outside Dom, or too many params.
         */
        template<Domain Dom = Domain::REAL, typename Func, typename Param = double>
        TaskHandle execute_logic_binary(TaskRecorder& recorder, const Tensor& A, const Tensor& B,
                                        Tensor& result, const char* name,
                                        SushiRuntime::Graph::OpID op_id, Func&& op_func,
                                        const std::vector<Param>& params = {})
        {
            require_storage(A, name, "A");
            require_storage(B, name, "B");
            require_storage(result, name, "result");
            require_contiguous(A, name, "A");
            require_contiguous(B, name, "B");
            require_contiguous(result, name, "result");
            require_same_shape(A, B, name, "A", "B");
            require_same_layout(A, B, name, "A", "B");
            require_same_shape(A, result, name, "A", "result");
            require_same_layout(A, result, name, "A", "result");
            SB_THROW_IF(A.dtype != B.dtype || A.dtype != result.dtype, "Data types must match for logic operation.");

            require_domain<Dom>(A.dtype, "Logic operation", name);

            if (result.num_elements == 0)
                return TaskHandle();

            int64_t size = A.num_elements;
            OperandSet operands;
            operands.reads(A).reads(B).writes(result);
            const bool vec_ok = Launch::vector_aligned(A.data(), B.data(), result.data());

            SushiRuntime::Graph::TaskMetadata meta;
            meta.name = name;
            meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
            meta.op_id = op_id;
            set_task_params(meta, params, name);

            return recorder.record(meta, operands, "Logic operation",
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
        }

        /**
         * @brief Records a ternary logic op `result[i] = op(cond[i], A[i], B[i])`, or nothing when result is empty.
         * @param cond Nonzero selects A, zero selects B.
         * @param result Must match the inputs in size and dtype.
         * @param op_func Callable `(auto c, auto a, auto b) -> auto` under execute_logic_unary's constraints.
         * @throws std::invalid_argument on a missing storage, a non-contiguous operand, a shape or layout mismatch, or under FORCE_REFERENCE.
         * @throws std::runtime_error on a dtype mismatch, a dtype outside Dom, or too many params.
         */
        template<Domain Dom = Domain::REAL, typename Func, typename Param = double>
        TaskHandle execute_logic_ternary(TaskRecorder& recorder, const Tensor& cond, const Tensor& A,
                                         const Tensor& B, Tensor& result, const char* name,
                                         SushiRuntime::Graph::OpID op_id, Func&& op_func,
                                         const std::vector<Param>& params = {})
        {
            require_storage(cond, name, "cond");
            require_storage(A, name, "A");
            require_storage(B, name, "B");
            require_storage(result, name, "result");
            require_contiguous(cond, name, "cond");
            require_contiguous(A, name, "A");
            require_contiguous(B, name, "B");
            require_contiguous(result, name, "result");
            require_same_shape(cond, A, name, "cond", "A");
            require_same_layout(cond, A, name, "cond", "A");
            require_same_shape(cond, B, name, "cond", "B");
            require_same_layout(cond, B, name, "cond", "B");
            require_same_shape(cond, result, name, "cond", "result");
            require_same_layout(cond, result, name, "cond", "result");
            SB_THROW_IF(cond.dtype != A.dtype || cond.dtype != B.dtype || cond.dtype != result.dtype, "Data types must match for logic operation.");

            require_domain<Dom>(cond.dtype, "Logic operation", name);

            if (result.num_elements == 0)
                return TaskHandle();

            int64_t size = cond.num_elements;
            OperandSet operands;
            operands.reads(cond).reads(A).reads(B).writes(result);
            const bool vec_ok = Launch::vector_aligned(cond.data(), A.data(), B.data(), result.data());

            SushiRuntime::Graph::TaskMetadata meta;
            meta.name = name;
            meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
            meta.op_id = op_id;
            set_task_params(meta, params, name);

            return recorder.record(meta, operands, "Logic operation",
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

        // No complex overloads follow; the ordering ops are Domain::REAL; see include/SushiBLAS/engine/logic/README.md.

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
