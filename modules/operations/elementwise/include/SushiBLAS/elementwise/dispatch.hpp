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
 * @brief Declares the elementwise executors of the dispatch seam.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <string>
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
#include <SushiBLAS/support/safe_math.hpp>
#include <SushiBLAS/functor_map/map_task.hpp>
#include <SushiBLAS/functor_map/read_binding.hpp>
#include <SushiBLAS/elementwise/wrapping_math.hpp>
#include <SushiBLAS/functor_map/scalar_binding.hpp>
#include <SushiRuntime/task/task_types.hpp>

namespace SushiBLAS
{
    namespace Dispatch
    {
        /**
         * @brief Records a unary in-place elementwise op `t[i] = op(t[i])`, or nothing when t is empty.
         * @tparam Dom Defaults to Domain::Real; a wider domain needs an op_func well-formed for it.
         * @param op_func Callable `(auto x) -> auto`, or a bind_scalars or bind_reads bundle of one.
         * @throws std::invalid_argument on an operand or declared-read violation, or too many params.
         * @throws UnsupportedExecution on a device-only tensor under ForceReference, or a dtype a tensor's device cannot compute in.
         * @see modules/operations/MATH_README.md
         */
        template<Domain Dom = Domain::Real, typename Func, typename Param = double>
        TaskHandle execute_unary_inplace(TaskRecorder& recorder, Tensor& t, const char* name,
                                         SushiRuntime::Graph::OpID op_id, Func&& op_func,
                                         std::initializer_list<Param> params = {})
        {
            require_storage(t, name, "t");
            require_device_dtype(t, name, "t");
            require_contiguous(t, name, "t");
            require_domain<Dom>(t.dtype, "Elementwise operation", name);
            Detail::require_declared_reads(op_func, t.dtype, name);

            if (t.num_elements == 0)
                return TaskHandle();

            OperandSet operands;
            operands.updates(t);
            Detail::declare_reads(op_func, operands, name);

            SushiRuntime::Graph::TaskMetadata meta;
            meta.name = name;
            meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
            meta.op_id = op_id;
            set_task_params(meta, params, name);

            void* ptr = t.data();
            return recorder.record_pair(meta, operands, "Elementwise operation",
                Detail::make_map_task<Dom>(t.dtype, "Elementwise operation", name,
                                           static_cast<std::size_t>(t.num_elements), ptr,
                                           Detail::task_functor(op_func),
                                           static_cast<const void*>(ptr)));
        }

        /**
         * @brief Records a binary elementwise op `C[i] = op(A[i], B[i])`, or nothing when C is empty.
         * @param C May alias an input.
         * @param op_func Callable `(auto x, auto y) -> auto`, or a bind_scalars or bind_reads bundle of one.
         * @throws std::invalid_argument on an operand or declared-read violation, or too many params.
         * @throws UnsupportedExecution on a device-only tensor under ForceReference, or a dtype a tensor's device cannot compute in.
         * @see modules/operations/MATH_README.md
         */
        template<Domain Dom = Domain::Real, typename Func, typename Param = double>
        TaskHandle execute_binary(TaskRecorder& recorder, const Tensor& A, const Tensor& B, Tensor& C,
                                  const char* name, SushiRuntime::Graph::OpID op_id, Func&& op_func,
                                  std::initializer_list<Param> params = {})
        {
            require_storage(A, name, "A");
            require_device_dtype(A, name, "A");
            require_storage(B, name, "B");
            require_device_dtype(B, name, "B");
            require_storage(C, name, "C");
            require_device_dtype(C, name, "C");
            require_contiguous(A, name, "A");
            require_contiguous(B, name, "B");
            require_contiguous(C, name, "C");
            require_same_shape(A, B, name, "A", "B");
            require_same_layout(A, B, name, "A", "B");
            require_same_shape(A, C, name, "A", "C");
            require_same_layout(A, C, name, "A", "C");
            require_dtype(B, A.dtype, name, "B");
            require_dtype(C, A.dtype, name, "C");

            require_domain<Dom>(A.dtype, "Elementwise operation", name);
            Detail::require_declared_reads(op_func, A.dtype, name);

            if (C.num_elements == 0)
                return TaskHandle();

            OperandSet operands;
            operands.reads(A).reads(B).writes(C);
            Detail::declare_reads(op_func, operands, name);

            SushiRuntime::Graph::TaskMetadata meta;
            meta.name = name;
            meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
            meta.op_id = op_id;
            set_task_params(meta, params, name);

            return recorder.record_pair(meta, operands, "Elementwise operation",
                Detail::make_map_task<Dom>(A.dtype, "Elementwise operation", name,
                                           static_cast<std::size_t>(A.num_elements), C.data(),
                                           Detail::task_functor(op_func), A.data(), B.data()));
        }

        /**
         * @brief Records a fused in-place op over N same-shaped tensors, or nothing when they are empty.
         * @param tensors None may be null; all must share one shape, layout and dtype, and none may overlap.
         * @param op_func Callable `(auto&...) -> void` over N lvalues, or a bind_scalars or bind_reads bundle of one.
         * @throws std::invalid_argument on an operand or declared-read violation, or too many params.
         * @throws UnsupportedExecution on a device-only tensor under ForceReference, or a dtype a tensor's device cannot compute in.
         * @see modules/operations/MATH_README.md
         */
        template<Domain Dom = Domain::Real, std::size_t N, typename Func, typename Param = double>
        TaskHandle execute_zip_inplace(TaskRecorder& recorder, const std::array<Tensor*, N>& tensors,
                                       const char* name, SushiRuntime::Graph::OpID op_id,
                                       Func&& op_func, std::initializer_list<Param> params = {})
        {
            static_assert(N >= 1, "A fused in-place elementwise op needs at least one tensor.");

            require_operand(tensors[0] != nullptr, name, "operand 0", "is null");
            require_storage(*tensors[0], name, "operand 0");
            require_device_dtype(*tensors[0], name, "operand 0");
            require_contiguous(*tensors[0], name, "operand 0");

            const int64_t size = tensors[0]->num_elements;
            const Core::DataType dtype = tensors[0]->dtype;

            for (std::size_t k = 1; k < N; ++k)
            {
                const std::string operand = "operand " + std::to_string(k);
                require_operand(tensors[k] != nullptr, name, operand.c_str(), "is null");
                require_storage(*tensors[k], name, operand.c_str());
                require_device_dtype(*tensors[k], name, operand.c_str());
                require_contiguous(*tensors[k], name, operand.c_str());
                require_same_shape(*tensors[k], *tensors[0], name, operand.c_str(), "operand 0");
                require_same_layout(*tensors[k], *tensors[0], name, operand.c_str(), "operand 0");
                require_dtype(*tensors[k], dtype, name, operand.c_str());
            }

            require_domain<Dom>(dtype, "Elementwise operation", name);
            Detail::require_declared_reads(op_func, dtype, name);

            if (size == 0)
                return TaskHandle();

            // Every operand is declared as both a read and a write, since the functor may touch any; see modules/operations/MATH_README.md.
            std::array<void*, N> ptrs{};
            OperandSet operands;
            for (std::size_t k = 0; k < N; ++k)
            {
                ptrs[k] = tensors[k]->data();
                operands.updates(*tensors[k]);
            }
            Detail::declare_reads(op_func, operands, name);

            SushiRuntime::Graph::TaskMetadata meta;
            meta.name = name;
            meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
            meta.op_id = op_id;
            set_task_params(meta, params, name);

            using Op = std::decay_t<decltype(Detail::task_functor(op_func))>;
            return recorder.record_pair(meta, operands, "Elementwise operation",
                Detail::ZipTask<Dom, Op, N>(dtype, "Elementwise operation", name,
                                            static_cast<std::size_t>(size), ptrs,
                                            Detail::task_functor(op_func)));
        }

        /**
         * @brief Returns `a + b`, wrapping modulo 2^N when T is an integer type.
         * @see modules/operations/MATH_README.md
         */
        template<typename T>
        inline T safe_ew_add(T a, T b)
        {
            return wrapping_add(a, b);
        }

        /**
         * @brief Returns `a - b`, wrapping modulo 2^N when T is an integer type.
         * @see modules/operations/MATH_README.md
         */
        template<typename T>
        inline T safe_ew_sub(T a, T b)
        {
            return wrapping_sub(a, b);
        }

        /**
         * @brief Returns `a * b`, wrapping modulo 2^N when T is an integer type.
         * @see modules/operations/MATH_README.md
         */
        template<typename T>
        inline T safe_ew_mul(T a, T b)
        {
            return wrapping_mul(a, b);
        }

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
