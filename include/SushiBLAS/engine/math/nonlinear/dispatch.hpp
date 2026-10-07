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
 * @brief Declares the activation executors of the dispatch seam.
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
#include <SushiBLAS/engine/math/safe_math.hpp>
#include <SushiRuntime/core/resource_region.hpp>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiBLAS
{
    namespace Dispatch
    {
        /**
         * @brief Records an activation's forward pass `t[i] = op(t[i])`, or nothing when t is
         * empty.
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
        TaskHandle execute_nonlinear_forward(TaskRecorder& recorder, Tensor& t, const char* name,
                                             SushiRuntime::Graph::OpID op_id, Func&& op_func,
                                             const std::vector<Param>& params = {})
        {
            require_storage(t, name, "t");
            require_contiguous(t, name, "t");
            require_domain<Dom>(t.dtype, "Nonlinear operation", name);

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

            return recorder.record(meta, operands, "Nonlinear operation",
                [size, ptr, vec_ok, dtype = t.dtype, op_func, name]
                (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
                {
                    SB_LOG_INFO("{} Forward: {} elements", name, size);
                    return q.submit([&](sycl::handler& h)
                    {
                        h.depends_on(deps);
                        const std::size_t n = static_cast<std::size_t>(size);
                        Dispatch::with_element_type<Dom>(dtype, "Nonlinear operation", name,
                            [&](auto tag)
                        {
                            using T = typename decltype(tag)::type;

                            Launch::map(h, q, n, vec_ok, op_func, (T*)ptr, (const T*)ptr);
                        });
                    });
                });
        }

        /**
         * @brief Records an activation's backward pass `dx[i] = op(dy[i], x[i])`, or nothing when dx is empty.
         * @param x The saved forward input or output the derivative is taken against.
         * @param op_func Callable `(auto dy, auto x) -> auto` under execute_nonlinear_forward's constraints.
         * @throws std::invalid_argument on a missing storage, a non-contiguous operand, a shape or layout mismatch, or under FORCE_REFERENCE.
         * @throws std::runtime_error on a dtype mismatch, a dtype outside Dom, or too many params.
         * @see include/SushiBLAS/engine/math/README.md
         */
        template<Domain Dom = Domain::REAL, typename Func, typename Param = double>
        TaskHandle execute_nonlinear_backward(TaskRecorder& recorder, const Tensor& dy,
                                              const Tensor& x, Tensor& dx, const char* name,
                                              SushiRuntime::Graph::OpID op_id, Func&& op_func,
                                              const std::vector<Param>& params = {})
        {
            require_storage(dy, name, "dy");
            require_storage(x, name, "x");
            require_storage(dx, name, "dx");
            require_contiguous(dy, name, "dy");
            require_contiguous(x, name, "x");
            require_contiguous(dx, name, "dx");
            require_same_shape(dy, x, name, "dy", "x");
            require_same_layout(dy, x, name, "dy", "x");
            require_same_shape(dy, dx, name, "dy", "dx");
            require_same_layout(dy, dx, name, "dy", "dx");
            SB_THROW_IF(dy.dtype != x.dtype || dy.dtype != dx.dtype,
                       "Tensor data types must match for backward operation.");

            require_domain<Dom>(x.dtype, "Nonlinear operation", name);

            if (dx.num_elements == 0)
                return TaskHandle();

            int64_t size = x.num_elements;
            OperandSet operands;
            operands.reads(dy).reads(x).writes(dx);
            const bool vec_ok = Launch::vector_aligned(dy.data(), x.data(), dx.data());

            SushiRuntime::Graph::TaskMetadata meta;
            meta.name = name;
            meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
            meta.op_id = op_id;
            set_task_params(meta, params, name);

            return recorder.record(meta, operands, "Nonlinear operation",
                [size, vec_ok, pDY_raw = dy.data(), pX_raw = x.data(), pDX_raw = dx.data(), dtype = x.dtype, op_func, name](sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
                {
                    SB_LOG_INFO("{} Backward: {} elements", name, size);
                    return q.submit([&](sycl::handler& h)
                    {
                        h.depends_on(deps);
                        const std::size_t n = static_cast<std::size_t>(size);
                        Dispatch::with_element_type<Dom>(dtype, "Nonlinear operation", name,
                            [&](auto tag)
                        {
                            using T = typename decltype(tag)::type;

                            Launch::map(h, q, n, vec_ok, op_func, (T*)pDX_raw, (const T*)pDY_raw, (const T*)pX_raw);
                        });
                    });
                });
        }

    } // namespace Dispatch
} // namespace SushiBLAS
