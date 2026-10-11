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

#include <complex>
#include <cstddef>
#include <initializer_list>
#include <sycl/sycl.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/engine/dispatch.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/task_handle.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiBLAS/functor_map/map_task.hpp>
#include <SushiBLAS/functor_map/read_binding.hpp>
#include <SushiRuntime/core/operation_id.hpp>

namespace SushiBLAS
{
    namespace Dispatch
    {
        /**
         * @brief Records a unary logic op `result[i] = op(x[i])`, or nothing when result is empty.
         * @param result Must match x in size and dtype.
         * @param op_func Callable `(auto x) -> auto`, or a bind_reads bundle of one: device-copyable, by-value captures, no allocation, no throw.
         * @throws std::invalid_argument on an operand or declared-read violation, or too many params.
         * @throws UnsupportedExecution on a device-only tensor under ForceReference, or a dtype a tensor's device cannot compute in.
         * @see modules/operations/logic/README.md
         */
        template<Domain Dom = Domain::Real, typename Func, typename Param = double>
        TaskHandle execute_logic_unary(TaskRecorder& recorder, const Tensor& x, Tensor& result,
                                       const char* name, SushiRuntime::OperationID op_id,
                                       Func&& op_func, std::initializer_list<Param> params = {})
        {
            require_storage(x, name, "x");
            require_device_dtype(x, name, "x");
            require_storage(result, name, "result");
            require_device_dtype(result, name, "result");
            require_contiguous(x, name, "x");
            require_contiguous(result, name, "result");
            require_same_shape(x, result, name, "x", "result");
            require_same_layout(x, result, name, "x", "result");
            require_dtype(result, x.dtype, name, "result");

            require_domain<Dom>(x.dtype, "Logic operation", name);
            Detail::require_declared_reads(op_func, x.dtype, name);

            if (result.num_elements == 0)
                return TaskHandle();

            OperandSet operands;
            operands.reads(x).writes(result);
            Detail::declare_reads(op_func, operands, name);

            SushiRuntime::Internal::Graph::TaskMetadata meta;
            meta.name = name;
            meta.task_type = SushiRuntime::Internal::Graph::TaskType::MathOp;
            meta.op_id = op_id;
            set_task_params(meta, params, name);

            return recorder.record_pair(meta, operands, "Logic operation",
                Detail::make_map_task<Dom>(x.dtype, "Logic operation", name,
                                           static_cast<std::size_t>(x.num_elements), result.data(),
                                           Detail::task_functor(op_func), x.data()));
        }

        /**
         * @brief Records a binary logic op `result[i] = op(A[i], B[i])`, or nothing when result is empty.
         * @tparam Dom Defaults to Domain::Real, because `<` and `>` have no meaning on complex numbers.
         * @param result Must match the inputs in size and dtype.
         * @param op_func Callable `(auto x, auto y) -> auto`, or a bind_reads bundle of one, under execute_logic_unary's constraints.
         * @throws std::invalid_argument on an operand or declared-read violation, or too many params.
         * @throws UnsupportedExecution on a device-only tensor under ForceReference, or a dtype a tensor's device cannot compute in.
         */
        template<Domain Dom = Domain::Real, typename Func, typename Param = double>
        TaskHandle execute_logic_binary(TaskRecorder& recorder, const Tensor& A, const Tensor& B,
                                        Tensor& result, const char* name,
                                        SushiRuntime::OperationID op_id, Func&& op_func,
                                        std::initializer_list<Param> params = {})
        {
            require_storage(A, name, "A");
            require_device_dtype(A, name, "A");
            require_storage(B, name, "B");
            require_device_dtype(B, name, "B");
            require_storage(result, name, "result");
            require_device_dtype(result, name, "result");
            require_contiguous(A, name, "A");
            require_contiguous(B, name, "B");
            require_contiguous(result, name, "result");
            require_same_shape(A, B, name, "A", "B");
            require_same_layout(A, B, name, "A", "B");
            require_same_shape(A, result, name, "A", "result");
            require_same_layout(A, result, name, "A", "result");
            require_dtype(B, A.dtype, name, "B");
            require_dtype(result, A.dtype, name, "result");

            require_domain<Dom>(A.dtype, "Logic operation", name);
            Detail::require_declared_reads(op_func, A.dtype, name);

            if (result.num_elements == 0)
                return TaskHandle();

            OperandSet operands;
            operands.reads(A).reads(B).writes(result);
            Detail::declare_reads(op_func, operands, name);

            SushiRuntime::Internal::Graph::TaskMetadata meta;
            meta.name = name;
            meta.task_type = SushiRuntime::Internal::Graph::TaskType::MathOp;
            meta.op_id = op_id;
            set_task_params(meta, params, name);

            return recorder.record_pair(meta, operands, "Logic operation",
                Detail::make_map_task<Dom>(A.dtype, "Logic operation", name,
                                           static_cast<std::size_t>(A.num_elements), result.data(),
                                           Detail::task_functor(op_func), A.data(), B.data()));
        }

        /**
         * @brief Records a ternary logic op `result[i] = op(cond[i], A[i], B[i])`, or nothing when result is empty.
         * @param cond Nonzero selects A, zero selects B.
         * @param result Must match the inputs in size and dtype.
         * @param op_func Callable `(auto c, auto a, auto b) -> auto`, or a bind_reads bundle of one, under execute_logic_unary's constraints.
         * @throws std::invalid_argument on an operand or declared-read violation, or too many params.
         * @throws UnsupportedExecution on a device-only tensor under ForceReference, or a dtype a tensor's device cannot compute in.
         */
        template<Domain Dom = Domain::Real, typename Func, typename Param = double>
        TaskHandle execute_logic_ternary(TaskRecorder& recorder, const Tensor& cond, const Tensor& A,
                                         const Tensor& B, Tensor& result, const char* name,
                                         SushiRuntime::OperationID op_id, Func&& op_func,
                                         std::initializer_list<Param> params = {})
        {
            require_storage(cond, name, "cond");
            require_device_dtype(cond, name, "cond");
            require_storage(A, name, "A");
            require_device_dtype(A, name, "A");
            require_storage(B, name, "B");
            require_device_dtype(B, name, "B");
            require_storage(result, name, "result");
            require_device_dtype(result, name, "result");
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
            require_dtype(A, cond.dtype, name, "A");
            require_dtype(B, cond.dtype, name, "B");
            require_dtype(result, cond.dtype, name, "result");

            require_domain<Dom>(cond.dtype, "Logic operation", name);
            Detail::require_declared_reads(op_func, cond.dtype, name);

            if (result.num_elements == 0)
                return TaskHandle();

            OperandSet operands;
            operands.reads(cond).reads(A).reads(B).writes(result);
            Detail::declare_reads(op_func, operands, name);

            SushiRuntime::Internal::Graph::TaskMetadata meta;
            meta.name = name;
            meta.task_type = SushiRuntime::Internal::Graph::TaskType::MathOp;
            meta.op_id = op_id;
            set_task_params(meta, params, name);

            return recorder.record_pair(meta, operands, "Logic operation",
                Detail::make_map_task<Dom>(cond.dtype, "Logic operation", name,
                                           static_cast<std::size_t>(cond.num_elements),
                                           result.data(), Detail::task_functor(op_func),
                                           cond.data(), A.data(), B.data()));
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

        // No complex overloads follow; the ordering ops are Domain::RealAndIntegral; see modules/operations/logic/README.md.

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
