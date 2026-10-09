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
#include <SushiBLAS/functor_map/scalar_binding.hpp>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiBLAS
{
    namespace Dispatch
    {
        /**
         * @brief Records an activation's forward pass `t[i] = op(t[i])`, or nothing when t is empty.
         * @tparam Dom Defaults to Domain::Real; a wider domain needs an op_func well-formed for it.
         * @param op_func Callable `(auto x) -> auto`, or a bind_scalars or bind_reads bundle of one.
         * @throws std::invalid_argument on an operand or declared-read violation, or too many params.
         * @throws UnsupportedExecution on a device-only tensor under ForceReference, or a dtype a tensor's device cannot compute in.
         * @see modules/operations/MATH_README.md
         */
        template<Domain Dom = Domain::Real, typename Func, typename Param = double>
        TaskHandle execute_nonlinear_forward(TaskRecorder& recorder, Tensor& t, const char* name,
                                             SushiRuntime::Graph::OpID op_id, Func&& op_func,
                                             std::initializer_list<Param> params = {})
        {
            require_storage(t, name, "t");
            require_device_dtype(t, name, "t");
            require_contiguous(t, name, "t");
            require_domain<Dom>(t.dtype, "Nonlinear operation", name);
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
            return recorder.record_pair(meta, operands, "Nonlinear operation",
                Detail::make_map_task<Dom>(t.dtype, "Nonlinear operation", name,
                                           static_cast<std::size_t>(t.num_elements), ptr,
                                           Detail::task_functor(op_func),
                                           static_cast<const void*>(ptr)));
        }

        /**
         * @brief Records an activation's backward pass `dx[i] = op(dy[i], x[i])`, or nothing when dx is empty.
         * @param x The saved forward input or output the derivative is taken against.
         * @param op_func Callable `(auto dy, auto x) -> auto`, or a bind_scalars or bind_reads bundle of one.
         * @throws std::invalid_argument on an operand or declared-read violation, or too many params.
         * @throws UnsupportedExecution on a device-only tensor under ForceReference, or a dtype a tensor's device cannot compute in.
         * @see modules/operations/MATH_README.md
         */
        template<Domain Dom = Domain::Real, typename Func, typename Param = double>
        TaskHandle execute_nonlinear_backward(TaskRecorder& recorder, const Tensor& dy,
                                              const Tensor& x, Tensor& dx, const char* name,
                                              SushiRuntime::Graph::OpID op_id, Func&& op_func,
                                              std::initializer_list<Param> params = {})
        {
            require_storage(dy, name, "dy");
            require_device_dtype(dy, name, "dy");
            require_storage(x, name, "x");
            require_device_dtype(x, name, "x");
            require_storage(dx, name, "dx");
            require_device_dtype(dx, name, "dx");
            require_contiguous(dy, name, "dy");
            require_contiguous(x, name, "x");
            require_contiguous(dx, name, "dx");
            require_same_shape(dy, x, name, "dy", "x");
            require_same_layout(dy, x, name, "dy", "x");
            require_same_shape(dy, dx, name, "dy", "dx");
            require_same_layout(dy, dx, name, "dy", "dx");
            require_dtype(x, dy.dtype, name, "x");
            require_dtype(dx, dy.dtype, name, "dx");

            require_domain<Dom>(x.dtype, "Nonlinear operation", name);
            Detail::require_declared_reads(op_func, x.dtype, name);

            if (dx.num_elements == 0)
                return TaskHandle();

            OperandSet operands;
            operands.reads(dy).reads(x).writes(dx);
            Detail::declare_reads(op_func, operands, name);

            SushiRuntime::Graph::TaskMetadata meta;
            meta.name = name;
            meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
            meta.op_id = op_id;
            set_task_params(meta, params, name);

            return recorder.record_pair(meta, operands, "Nonlinear operation",
                Detail::make_map_task<Dom>(x.dtype, "Nonlinear operation", name,
                                           static_cast<std::size_t>(x.num_elements), dx.data(),
                                           Detail::task_functor(op_func), dy.data(), x.data()));
        }

    } // namespace Dispatch
} // namespace SushiBLAS
