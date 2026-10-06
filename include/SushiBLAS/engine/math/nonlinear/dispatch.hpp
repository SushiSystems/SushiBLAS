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
#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/tensor.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/engine/dispatch.hpp>
#include <SushiBLAS/engine/launch_1d.hpp>
#include <SushiBLAS/engine/math/safe_math.hpp>
#include <SushiRuntime/core/resource_region.hpp>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiBLAS
{
    namespace Dispatch
    {
        /**
         * @brief Records an activation's forward pass in place: `t[i] = op(t[i])`.
         * @param op_func Callable `(auto x) -> auto`: device-copyable, by-value captures, no allocation, no throw.
         * @throws std::runtime_error if the dtype is unrecognized or params exceeds the inline capacity.
         * @see include/SushiBLAS/engine/math/README.md
         */
        template<Domain Dom = Domain::REAL_AND_COMPLEX, typename Func>
        sycl::event execute_nonlinear_forward(Engine& engine, Tensor& t, const char* name, SushiRuntime::Graph::OpID op_id, Func&& op_func, const std::vector<float>& params = {})
        {
            require_domain<Dom>(t.dtype, "Nonlinear operation", name);

            int64_t size = t.num_elements;
            void* ptr = t.data_or_null();
            const bool vec_ok = Launch::is_vector_aligned(ptr);
            std::vector<SushiRuntime::Core::ResourceRegion> rw = ptr ? std::vector<SushiRuntime::Core::ResourceRegion>{t.region()} : std::vector<SushiRuntime::Core::ResourceRegion>{};

            SushiRuntime::Graph::TaskMetadata meta;
            meta.name = name;
            meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
            meta.op_id = op_id;
            set_task_params(meta, params, name);

            engine.get_graph().add_task(meta, rw, rw,
                [size, ptr, vec_ok, dtype = t.dtype, op_func, name](sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
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
            return sycl::event();
        }

        /**
         * @brief Records an activation's backward pass: `dx[i] = op(dy[i], x[i])`.
         * @param x The saved forward input or output the derivative is taken against.
         * @param op_func Callable `(auto dy, auto x) -> auto` under execute_nonlinear_forward's constraints.
         * @throws std::runtime_error on a size or dtype mismatch, an unrecognized dtype, or too many params.
         * @see include/SushiBLAS/engine/math/README.md
         */
        template<Domain Dom = Domain::REAL_AND_COMPLEX, typename Func>
        sycl::event execute_nonlinear_backward(Engine& engine, const Tensor& dy, const Tensor& x, Tensor& dx, const char* name, SushiRuntime::Graph::OpID op_id, Func&& op_func, const std::vector<float>& params = {})
        {
            SB_THROW_IF(dy.num_elements != x.num_elements || dy.num_elements != dx.num_elements,
                       "Tensor sizes must match for backward operation.");
            SB_THROW_IF(dy.dtype != x.dtype || dy.dtype != dx.dtype,
                       "Tensor data types must match for backward operation.");

            require_domain<Dom>(x.dtype, "Nonlinear operation", name);

            int64_t size = x.num_elements;
            std::vector<SushiRuntime::Core::ResourceRegion> reads = {dy.region(), x.region()};
            std::vector<SushiRuntime::Core::ResourceRegion> writes = {dx.region()};
            const bool vec_ok = Launch::vector_aligned(dy.data(), x.data(), dx.data());

            SushiRuntime::Graph::TaskMetadata meta;
            meta.name = name;
            meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
            meta.op_id = op_id;
            set_task_params(meta, params, name);

            engine.get_graph().add_task(meta, reads, writes,
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
            return sycl::event();
        }

    } // namespace Dispatch
} // namespace SushiBLAS
