/****************************************************************/
/* dispatch.hpp                                                 */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

#pragma once

#include <array>
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
            require_domain<Dom>(t.dtype, "Elementwise operation", name);

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
            SB_THROW_IF(A.num_elements != B.num_elements || A.num_elements != C.num_elements,
                    "Tensor sizes must match for elementwise operation.");
            SB_THROW_IF(A.dtype != B.dtype || A.dtype != C.dtype,
                    "Tensor data types must match for elementwise operation.");

            require_domain<Dom>(A.dtype, "Elementwise operation", name);

            int64_t size = A.num_elements;
            std::vector<SushiRuntime::Core::ResourceRegion> reads = {A.region(), B.region()};
            std::vector<SushiRuntime::Core::ResourceRegion> writes = {C.region()};
            const bool vec_ok = Launch::vector_aligned(A.data(), B.data(), C.data());

            SushiRuntime::Graph::TaskMetadata meta;
            meta.name = name;
            meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
            meta.op_id = op_id;
            set_task_params(meta, params, name);

            engine.get_graph().add_task(meta, reads, writes,
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
            return sycl::event();
        }

        /**
         * @brief Records a fused in-place elementwise op over N same-shaped tensors:
         * `op(t0[i], t1[i], ..., tN-1[i])` for every element, every slot writable.
         *
         * The variadic sibling of @ref execute_unary_inplace and @ref execute_binary, and
         * the seam a consumer reaches for when the thing it wants to run is not "one output
         * from some inputs" but "a group of tensors advanced together". @p op_func receives
         * an **lvalue reference** to each tensor's element, so it may read and write any of
         * them in a single pass.
         *
         * That is what turns a whole optimiser step into one kernel. An AdamW update issues
         * roughly eleven launches per parameter when it is spelled as separate ops - decay
         * the parameter, scale each moment, accumulate each moment, two bias corrections,
         * the epsilon, the division, the step - and every one of them re-reads and rewrites
         * state that is already in registers. Written through this seam it is one launch and
         * one pass over memory:
         *
         * @code
         * // theta, g, m, v and ema advanced together, in one kernel.
         * Dispatch::execute_zip_inplace(engine, std::array{&theta, &g, &m, &v, &ema},
         *     "optim.adamw", "optim.adamw"_op,
         *     [b1, b2, lr, wd, eps, inv_bc1, inv_bc2, decay]
         *     (auto& theta_i, auto& g_i, auto& m_i, auto& v_i, auto& ema_i)
         *     {
         *         using T = std::decay_t<decltype(theta_i)>;
         *         m_i = T(b1) * m_i + T(1 - b1) * g_i;
         *         v_i = T(b2) * v_i + T(1 - b2) * (g_i * g_i);
         *         const T step = (m_i * T(inv_bc1))
         *                      / (Dispatch::safe_sqrt(v_i * T(inv_bc2)) + T(eps));
         *         theta_i = theta_i * T(1 - lr * wd) - T(lr) * step;
         *         ema_i   = T(decay) * ema_i + T(1 - decay) * theta_i;
         *     });
         * @endcode
         *
         * **Why `std::array<Tensor*, N>` and not a pack or a span.** A pack would have to be
         * the trailing parameter, which would push `name`, `op_id`, `op_func` and the
         * defaulted `params` in front of the tensors and break the argument order every
         * other executor in this family uses. A `std::span` would make N a run-time value,
         * and the kernel needs it at compile time: it stages one register per operand per
         * element and expands the call to @p op_func through an `index_sequence`. The array
         * also says the true thing about these operands - that they are one homogeneous
         * group, not a list of separately-named roles - and `std::array{&a, &b, &c, &d, &e}`
         * stays legible at the N=5 the example above needs. Pointers rather than references
         * only because a reference cannot be an array element.
         *
         * **Graph bookkeeping.** All N tensors are declared as both reads and writes, since
         * the functor is free to touch any of them and this layer cannot see which. The
         * regions come from `Tensor::region()`, which reports a precise byte interval, so
         * two disjoint views of one arena still do not serialise against each other.
         *
         * @tparam Dom Which element types to instantiate the functor for. Defaults to
         *         Domain::REAL; pass Domain::REAL_AND_COMPLEX only when `op_func` is
         *         well-formed for `std::complex` (see safe_math.hpp).
         * @tparam N How many tensors are updated together, deduced.
         * @tparam Func The functor type, deduced.
         * @param engine The engine whose task graph the op is recorded on.
         * @param tensors The tensors to advance together, in the order @p op_func expects
         *        them. None may be null; all must agree on element count and dtype.
         * @param name Human-readable op name, used for logging and failure messages.
         * @param op_id Stable operation identifier for the task graph.
         * @param op_func Callable `(auto&...) -> void` taking exactly N lvalue references,
         *        invoked once per element index on device.
         * @param params Optional scalar parameters recorded in the task metadata.
         * @return An empty sycl::event; the work is deferred until `Engine::execute()`.
         *
         * @warning Same device-kernel constraints as @ref execute_binary: `op_func` is
         *          inlined into the kernel, so it must be device-copyable, capture by value,
         *          hold no host pointers, allocate nothing and throw nothing.
         * @warning The N tensors must not overlap one another. Each element index is staged
         *          into registers, handed to @p op_func, and written back, so a write
         *          through one operand is not visible to a second operand that happens to
         *          name the same address. Disjoint views of one arena are fine; two views of
         *          the same bytes are not.
         * @throws std::runtime_error if any tensor is null, if the tensors disagree on
         *         element count or dtype (the message names which one disagreed), if the
         *         dtype falls outside @p Dom, or if @p params exceeds the metadata's inline
         *         capacity.
         */
        template<Domain Dom = Domain::REAL, std::size_t N, typename Func>
        sycl::event execute_zip_inplace(Engine& engine, const std::array<Tensor*, N>& tensors, const char* name, SushiRuntime::Graph::OpID op_id, Func&& op_func, const std::vector<float>& params = {})
        {
            static_assert(N >= 1, "A fused in-place elementwise op needs at least one tensor.");

            SB_THROW_IF(tensors[0] == nullptr, "Fused elementwise operation '{}': operand 0 is null.", name);

            const int64_t size = tensors[0]->num_elements;
            const Core::DataType dtype = tensors[0]->dtype;

            for (std::size_t k = 1; k < N; ++k)
            {
                SB_THROW_IF(tensors[k] == nullptr, "Fused elementwise operation '{}': operand {} is null.", name, k);
                SB_THROW_IF(tensors[k]->num_elements != size,
                            "Fused elementwise operation '{}': operand {} holds {} elements but operand 0 holds {}; "
                            "every operand of a fused in-place update must have the same element count.",
                            name, k, tensors[k]->num_elements, size);
                SB_THROW_IF(tensors[k]->dtype != dtype,
                            "Fused elementwise operation '{}': operand {} has dtype {} but operand 0 has dtype {}; "
                            "every operand of a fused in-place update must share one dtype.",
                            name, k, Core::to_string(tensors[k]->dtype), Core::to_string(dtype));
            }

            require_domain<Dom>(dtype, "Elementwise operation", name);

            // Every operand is both a read and a write: the functor may touch any of them,
            // and nothing at this layer can tell which ones it actually did.
            std::array<void*, N> ptrs{};
            std::vector<SushiRuntime::Core::ResourceRegion> rw;
            rw.reserve(N);
            bool vec_ok = true;
            for (std::size_t k = 0; k < N; ++k)
            {
                ptrs[k] = tensors[k]->data();
                rw.push_back(tensors[k]->region());
                vec_ok = vec_ok && Launch::is_vector_aligned(ptrs[k]);
            }

            SushiRuntime::Graph::TaskMetadata meta;
            meta.name = name;
            meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
            meta.op_id = op_id;
            set_task_params(meta, params, name);

            engine.get_graph().add_task(meta, rw, rw,
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
