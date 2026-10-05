/****************************************************************/
/* dispatch.hpp                                                 */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

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
