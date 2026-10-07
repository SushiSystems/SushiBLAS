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
#include <SushiBLAS/engine/math/safe_math.hpp>
#include <SushiBLAS/engine/math/elementwise/wrapping_math.hpp>
#include <SushiRuntime/core/resource_region.hpp>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiBLAS
{
    namespace Dispatch
    {
        /**
         * @brief Records a unary in-place elementwise op `t[i] = op(t[i])`, or nothing when t
         * is empty.
         * @tparam Dom Defaults to Domain::REAL; REAL_AND_COMPLEX needs an op_func well-formed
         * for it.
         * @param op_func Callable `(auto x) -> auto`: device-copyable, by-value captures, no
         * allocation, no throw.
         * @throws std::invalid_argument if t holds elements without storage or is not contiguous,
         * or under FORCE_REFERENCE.
         * @throws std::runtime_error if the dtype falls outside Dom or params exceeds the inline
         * capacity.
         * @see include/SushiBLAS/engine/math/README.md
         */
        template<Domain Dom = Domain::REAL, typename Func, typename Param = double>
        TaskHandle execute_unary_inplace(TaskRecorder& recorder, Tensor& t, const char* name,
                                         SushiRuntime::Graph::OpID op_id, Func&& op_func,
                                         const std::vector<Param>& params = {})
        {
            require_storage(t, name, "t");
            require_contiguous(t, name, "t");
            require_domain<Dom>(t.dtype, "Elementwise operation", name);

            if (t.num_elements == 0)
                return TaskHandle();

            int64_t size = t.num_elements;
            void* ptr = t.data();
            const bool vec_ok = Launch::is_vector_aligned(ptr);
            OperandSet operands;
            operands.updates(t);

            SushiRuntime::Graph::TaskMetadata meta;
            meta.name = name;
            meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
            meta.op_id = op_id;
            set_task_params(meta, params, name);

            return recorder.record(meta, operands, "Elementwise operation",
                [size, ptr, vec_ok, dtype = t.dtype, op_func, name]
                (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
                {
                    SB_LOG_INFO("Elementwise {} (In-place): {} elements", name, size);
                    return q.submit([&](sycl::handler& h)
                    {
                        h.depends_on(deps);
                        const std::size_t n = static_cast<std::size_t>(size);
                        Dispatch::with_element_type<Dom>(dtype, "Elementwise operation", name,
                            [&](auto tag)
                        {
                            using T = typename decltype(tag)::type;

                            Launch::map(h, q, n, vec_ok, op_func, (T*)ptr, (const T*)ptr);
                        });
                    });
                });
        }

        /**
         * @brief Records a binary elementwise op `C[i] = op(A[i], B[i])`, or nothing when C is empty.
         * @tparam Dom Defaults to Domain::REAL; REAL_AND_COMPLEX needs an op_func well-formed for it.
         * @param C May alias an input.
         * @param op_func Callable `(auto x, auto y) -> auto` under execute_unary_inplace's constraints.
         * @throws std::invalid_argument on a missing storage, a non-contiguous operand, a shape or layout mismatch, or under FORCE_REFERENCE.
         * @throws std::runtime_error on a dtype mismatch, a dtype outside Dom, or too many params.
         * @see include/SushiBLAS/engine/math/README.md
         */
        template<Domain Dom = Domain::REAL, typename Func, typename Param = double>
        TaskHandle execute_binary(TaskRecorder& recorder, const Tensor& A, const Tensor& B, Tensor& C,
                                  const char* name, SushiRuntime::Graph::OpID op_id, Func&& op_func,
                                  const std::vector<Param>& params = {})
        {
            require_storage(A, name, "A");
            require_storage(B, name, "B");
            require_storage(C, name, "C");
            require_contiguous(A, name, "A");
            require_contiguous(B, name, "B");
            require_contiguous(C, name, "C");
            require_same_shape(A, B, name, "A", "B");
            require_same_layout(A, B, name, "A", "B");
            require_same_shape(A, C, name, "A", "C");
            require_same_layout(A, C, name, "A", "C");
            SB_THROW_IF(A.dtype != B.dtype || A.dtype != C.dtype,
                    "Tensor data types must match for elementwise operation.");

            require_domain<Dom>(A.dtype, "Elementwise operation", name);

            if (C.num_elements == 0)
                return TaskHandle();

            int64_t size = A.num_elements;
            OperandSet operands;
            operands.reads(A).reads(B).writes(C);
            const bool vec_ok = Launch::vector_aligned(A.data(), B.data(), C.data());

            SushiRuntime::Graph::TaskMetadata meta;
            meta.name = name;
            meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
            meta.op_id = op_id;
            set_task_params(meta, params, name);

            return recorder.record(meta, operands, "Elementwise operation",
                [size, vec_ok, pA_raw = A.data(), pB_raw = B.data(), pC_raw = C.data(), dtype = A.dtype, op_func, name](sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
                {
                    SB_LOG_INFO("Elementwise {}: {} elements", name, size);
                    return q.submit([&](sycl::handler& h)
                    {
                        h.depends_on(deps);
                        const std::size_t n = static_cast<std::size_t>(size);
                        Dispatch::with_element_type<Dom>(dtype, "Elementwise operation", name,
                            [&](auto tag)
                        {
                            using T = typename decltype(tag)::type;

                            Launch::map(h, q, n, vec_ok, op_func, (T*)pC_raw, (const T*)pA_raw, (const T*)pB_raw);
                        });
                    });
                });
        }

        /**
         * @brief Records a fused in-place op over N same-shaped tensors, or nothing when they are empty.
         * @param tensors None may be null; all must share one shape, layout and dtype, and none may overlap.
         * @param op_func Callable `(auto&...) -> void` taking N lvalue references.
         * @throws std::invalid_argument on a missing storage, a non-contiguous operand, a shape or layout mismatch, or under FORCE_REFERENCE.
         * @throws std::runtime_error on a null tensor, a dtype mismatch, or a dtype outside Dom.
         * @see include/SushiBLAS/engine/math/README.md
         */
        template<Domain Dom = Domain::REAL, std::size_t N, typename Func, typename Param = double>
        TaskHandle execute_zip_inplace(TaskRecorder& recorder, const std::array<Tensor*, N>& tensors,
                                       const char* name, SushiRuntime::Graph::OpID op_id,
                                       Func&& op_func, const std::vector<Param>& params = {})
        {
            static_assert(N >= 1, "A fused in-place elementwise op needs at least one tensor.");

            SB_THROW_IF(tensors[0] == nullptr, "Fused elementwise operation '{}': operand 0 is null.", name);
            require_storage(*tensors[0], name, "operand 0");
            require_contiguous(*tensors[0], name, "operand 0");

            const int64_t size = tensors[0]->num_elements;
            const Core::DataType dtype = tensors[0]->dtype;

            for (std::size_t k = 1; k < N; ++k)
            {
                SB_THROW_IF(tensors[k] == nullptr, "Fused elementwise operation '{}': operand {} is null.", name, k);
                const std::string operand = "operand " + std::to_string(k);
                require_storage(*tensors[k], name, operand.c_str());
                require_contiguous(*tensors[k], name, operand.c_str());
                require_same_shape(*tensors[k], *tensors[0], name, operand.c_str(), "operand 0");
                require_same_layout(*tensors[k], *tensors[0], name, operand.c_str(), "operand 0");
                SB_THROW_IF(tensors[k]->dtype != dtype,
                            "Fused elementwise operation '{}': operand {} has dtype {} but operand 0 has dtype {}; "
                            "every operand of a fused in-place update must share one dtype.",
                            name, k, Core::to_string(tensors[k]->dtype), Core::to_string(dtype));
            }

            require_domain<Dom>(dtype, "Elementwise operation", name);

            if (size == 0)
                return TaskHandle();

            // Every operand is declared as both a read and a write, since the functor may touch any; see include/SushiBLAS/engine/math/README.md.
            std::array<void*, N> ptrs{};
            OperandSet operands;
            bool vec_ok = true;
            for (std::size_t k = 0; k < N; ++k)
            {
                ptrs[k] = tensors[k]->data();
                operands.updates(*tensors[k]);
                vec_ok = vec_ok && Launch::is_vector_aligned(ptrs[k]);
            }

            SushiRuntime::Graph::TaskMetadata meta;
            meta.name = name;
            meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
            meta.op_id = op_id;
            set_task_params(meta, params, name);

            return recorder.record(meta, operands, "Elementwise operation",
                [size, ptrs, vec_ok, dtype, op_func, name](sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
                {
                    SB_LOG_INFO("Elementwise {} (Fused in-place, {} operands): {} elements", name, N, size);
                    return q.submit([&](sycl::handler& h)
                    {
                        h.depends_on(deps);
                        const std::size_t n = static_cast<std::size_t>(size);
                        Dispatch::with_element_type<Dom>(dtype, "Elementwise operation", name,
                            [&](auto tag)
                        {
                            using T = typename decltype(tag)::type;

                            Launch::zip_inplace(h, q, n, vec_ok, Launch::typed_ptrs<T>(ptrs), op_func);
                        });
                    });
                });
        }

        /**
         * @brief Returns `a + b`, wrapping modulo 2^N when T is an integer type.
         * @see include/SushiBLAS/engine/math/README.md
         */
        template<typename T>
        inline T safe_ew_add(T a, T b)
        {
            return wrapping_add(a, b);
        }

        /**
         * @brief Returns `a - b`, wrapping modulo 2^N when T is an integer type.
         * @see include/SushiBLAS/engine/math/README.md
         */
        template<typename T>
        inline T safe_ew_sub(T a, T b)
        {
            return wrapping_sub(a, b);
        }

        /**
         * @brief Returns `a * b`, wrapping modulo 2^N when T is an integer type.
         * @see include/SushiBLAS/engine/math/README.md
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
