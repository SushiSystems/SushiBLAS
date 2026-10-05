/****************************************************************/
/* dispatch.hpp                                                 */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file dispatch.hpp
 * @brief Declares the elementwise executors of the dispatch seam.
 * @author Mustafa Garip
 */

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
         * @tparam Dom Defaults to Domain::REAL; REAL_AND_COMPLEX needs an op_func well-formed for it.
         * @param op_func Callable `(auto x) -> auto`: device-copyable, by-value captures, no allocation, no throw.
         * @throws std::runtime_error if the dtype falls outside Dom or params exceeds the inline capacity.
         * @see include/SushiBLAS/engine/math/README.md
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
         * @tparam Dom Defaults to Domain::REAL; REAL_AND_COMPLEX needs an op_func well-formed for it.
         * @param C May alias an input.
         * @param op_func Callable `(auto x, auto y) -> auto` under execute_unary_inplace's constraints.
         * @throws std::runtime_error on a size or dtype mismatch, a dtype outside Dom, or too many params.
         * @see include/SushiBLAS/engine/math/README.md
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
         * @brief Records a fused in-place op over N same-shaped tensors, every slot writable.
         * @param tensors None may be null; all must agree on element count and dtype and none may overlap.
         * @param op_func Callable `(auto&...) -> void` taking N lvalue references.
         * @throws std::runtime_error on a null tensor, a count or dtype mismatch, or a dtype outside Dom.
         * @see include/SushiBLAS/engine/math/README.md
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

            // Every operand is declared as both a read and a write, since the functor may touch any; see include/SushiBLAS/engine/math/README.md.
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
