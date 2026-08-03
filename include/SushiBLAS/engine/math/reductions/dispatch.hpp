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
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

#pragma once

#include <vector>
#include <cstddef>
#include <sycl/sycl.hpp>
#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/tensor.hpp>
#include <SushiBLAS/storage.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/engine/dispatch.hpp>
#include <SushiBLAS/engine/reduce_1d.hpp>
#include <SushiBLAS/engine/device_policy.hpp>
#include <SushiRuntime/core/resource_region.hpp>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiBLAS
{
    namespace Dispatch
    {
        /**
         * @struct ReductionScratch
         * @brief A device buffer a reduction may stage per-work-group partials in.
         *
         * Handed to the dispatcher instead of an allocator, because the buffer is
         * obtained **once, at task-record time** and captured by the task - not
         * allocated per launch. That is the right trade specifically because this
         * library is compile-once/replay: a `sycl::malloc_device` plus a `host_task`
         * to free it would add two submissions to every reduction on every replay of
         * the graph, which is exactly the launch cost the last two commits existed to
         * remove. A buffer taken when the task is recorded costs nothing on any
         * subsequent replay.
         *
         * @tparam T The element type, matching the tensor being reduced.
         */
        template<typename T>
        struct ReductionScratch
        {
            /** @brief The partials buffer, or null when none was requested. */
            T* data = nullptr;

            /** @brief How many values @ref data can hold; 0 when there is none. */
            std::size_t capacity = 0;
        };

        /**
         * @brief Records a whole-tensor reduction to a scalar result tensor.
         *
         * The generic seam behind every `ReductionOps` entry point. Unlike the
         * elementwise and logic families, the caller supplies the entire command group
         * rather than a per-element functor: a reduction's shape is the caller's choice
         * and only the graph bookkeeping, the scratch and the dtype switch are shared.
         * Callers that want the library's own deterministic shape should reach for
         * @ref execute_folding_reduction instead, which is this function with that
         * shape already filled in.
         *
         * @tparam Dispatcher The command-group builder type, deduced.
         * @param engine The engine whose task graph the op is recorded on.
         * @param t The tensor to reduce.
         * @param result The scalar (single-element) output tensor; must share `t`'s dtype.
         * @param name Human-readable op name, used for logging and failure messages.
         * @param op_id Stable operation identifier for the task graph.
         * @param dispatch A generic callable (typically a templated lambda) with the
         *        signature `(sycl::queue& q, const auto* pT, auto* pR, int64_t size,
         *        ReductionScratch<...> scratch, const std::vector<sycl::event>& deps)
         *        -> sycl::event`, instantiated once per supported element type.
         * @param scratch_elements How many per-work-group partials the dispatcher wants
         *        room for. Zero - the default - allocates nothing and hands the
         *        dispatcher an empty @ref ReductionScratch, which is what every
         *        single-pass reduction (argmax, var, logsumexp, ...) wants.
         *        @ref SushiBLAS::Reduce::planned_partials is how to compute it.
         * @return An empty sycl::event; the work is deferred until `Engine::execute()`.
         *
         * @warning This is a low-level generic seam. Whatever `dispatch` submits runs on
         *          the device, so any lambda it nests must be device-copyable, capture by
         *          value, hold no host pointers, allocate nothing and throw nothing.
         *          `dispatch` itself runs on the host at execution time and owns the
         *          `depends_on(deps)` call - dropping it silently breaks the graph's
         *          ordering guarantees.
         * @throws std::runtime_error if `result` is not scalar, if the dtypes disagree, or
         *         if the dtype is one reductions do not support (HALF, FLOAT32 and FLOAT64
         *         are; the complex types are not, since a reduction's operators need an
         *         ordering and an identity that `std::complex` does not supply).
         */
        template<typename Dispatcher>
        sycl::event execute_reduction(Engine& engine, const Tensor& t, Tensor& result, const char* name, SushiRuntime::Graph::OpID op_id, Dispatcher&& dispatch, std::size_t scratch_elements = 0)
        {
            SB_THROW_IF(result.num_elements != 1, "Result tensor for reduction '{}' must be a scalar (1 element).", name);
            SB_THROW_IF(t.dtype != result.dtype, "Data types must match for reduction operations.");

            int64_t size = t.num_elements;
            void* read_T = t.data();
            void* write_R = result.data();

            // `Tensor::region()`, never the bare `data()` pointer. A region is keyed on
            // the *allocation base* and narrowed to the view's byte interval; a bare
            // pointer converts to the whole-buffer region of whatever it points at. For a
            // tensor at offset zero those are the same thing, which is why this went
            // unnoticed - but for two views inside one arena the bare pointers are two
            // different bases, `ResourceRegion::overlaps` returns false, and the tracker
            // emits no edge at all. Not a wrong edge: none. A reduction then races the op
            // that produced its input, and the result is a plausible wrong number with no
            // error anywhere. SushiAI's first arena-packed training run lost exactly this
            // way (a loss of -57 where ln 6 was expected).
            std::vector<SushiRuntime::Core::ResourceRegion> reads = {t.region()};
            std::vector<SushiRuntime::Core::ResourceRegion> writes = {result.region()};

            // Scratch is a declared resource, not a hidden one. Each recorded reduction
            // owns its own allocation today, so there is nothing for the tracker to order
            // it against - but declaring it is what keeps that true if the scratch ever
            // comes from a shared pool, and it is described by a precise region for the
            // same reason the operands are.
            const std::size_t bpe = Core::bytes_per_element(t.dtype);
            SushiRuntime::sushi_ptr<Storage> scratch;
            void* scratch_ptr = nullptr;

            if (scratch_elements > 0)
            {
                scratch = SushiRuntime::make_sushi<Storage>(
                    engine.get_context().get_allocator(), scratch_elements * bpe,
                    SushiRuntime::Memory::Strategies::Device::instance());
                scratch_ptr = scratch->allocation_base();
                writes.push_back({scratch_ptr, 0, scratch_elements * bpe});
            }

            SushiRuntime::Graph::TaskMetadata meta;
            meta.name = name;
            meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
            meta.op_id = op_id;

            switch (t.dtype)
            {
                case Core::DataType::HALF:
                    engine.get_graph().add_task(meta, reads, writes,
                        [size, pT=(sycl::half*)read_T, pR=(sycl::half*)write_R, scratch,
                         pS=(sycl::half*)scratch_ptr, scratch_elements, dispatch, name]
                        (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
                        {
                            SB_LOG_INFO("Reduction {}: {} elements", name, size);
                            return dispatch(q, pT, pR, size,
                                            ReductionScratch<sycl::half>{pS, scratch_elements}, deps);
                        });
                    break;
                case Core::DataType::FLOAT32:
                    engine.get_graph().add_task(meta, reads, writes,
                        [size, pT=(float*)read_T, pR=(float*)write_R, scratch,
                         pS=(float*)scratch_ptr, scratch_elements, dispatch, name]
                        (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
                        {
                            SB_LOG_INFO("Reduction {}: {} elements", name, size);
                            return dispatch(q, pT, pR, size,
                                            ReductionScratch<float>{pS, scratch_elements}, deps);
                        });
                    break;
                case Core::DataType::FLOAT64:
                    engine.get_graph().add_task(meta, reads, writes,
                        [size, pT=(double*)read_T, pR=(double*)write_R, scratch,
                         pS=(double*)scratch_ptr, scratch_elements, dispatch, name]
                        (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
                        {
                            SB_LOG_INFO("Reduction {}: {} elements", name, size);
                            return dispatch(q, pT, pR, size,
                                            ReductionScratch<double>{pS, scratch_elements}, deps);
                        });
                    break;
                default:
                    SB_THROW_IF(true, "Reduction '{}' does not support dtype {}.", name, Core::to_string(t.dtype));
                    break;
            }
            return sycl::event();
        }

        /**
         * @brief Records a whole-tensor reduction using the library's own deterministic
         * folding shape: `result = finalize(fold(transform(t[i])), n)`.
         *
         * The seam every order-sensitive reduction in the library goes through -
         * `sum`, `mean`, `asum`, `nrm2` and `prod` are each one call to this, differing
         * only in @p Accum, @p transform and @p finalize. That is the point: the part of
         * a reduction that is hard to get right is the *combination order*, and putting
         * it in five places is five chances to get it wrong. See
         * <SushiBLAS/engine/reduce_1d.hpp> for the shape and for why `sycl::reduction`
         * and `sycl::reduce_over_group` are both unusable here.
         *
         * It sizes and requests the partials scratch itself, from the device the tensor's
         * storage lives on, so the caller never sees an allocator.
         *
         * @code
         * // A root-mean-square reduction a consumer could write, deterministic and
         * // compensated, in one call and one kernel:
         * Dispatch::execute_folding_reduction<Reduce::CompensatedSum>(
         *     engine, t, result, "nn.rms", "nn.rms"_op,
         *     [](auto x) { return x * x; },
         *     [](auto v, std::size_t n) { return sycl::sqrt(v / decltype(v)(n)); });
         * @endcode
         *
         * @tparam Accum The accumulation policy: @ref SushiBLAS::Reduce::CompensatedSum
         *         for anything additive, @ref SushiBLAS::Reduce::RunningProduct for a
         *         product.
         * @tparam Transform The per-element transform type, deduced.
         * @tparam Finalize The finalizer type, deduced.
         * @param engine The engine whose task graph the op is recorded on.
         * @param t The tensor to reduce.
         * @param result The scalar (single-element) output tensor; must share `t`'s dtype.
         * @param name Human-readable op name, used for logging and failure messages.
         * @param op_id Stable operation identifier for the task graph.
         * @param transform Callable `(auto x) -> auto`, applied to each element before it
         *        is folded in. Pass the identity for a plain sum.
         * @param finalize Callable `(auto value, std::size_t n) -> auto`, applied once to
         *        the folded result. Pass the identity when there is nothing to do; it is
         *        folded into the last kernel either way, so `mean` and `nrm2` cost no more
         *        launches than `sum`.
         * @return An empty sycl::event; the work is deferred until `Engine::execute()`.
         *
         * @warning @p transform and @p finalize are inlined into a SYCL device kernel and
         *          carry the same constraints as every other functor in this family:
         *          device-copyable, capture by value, no host pointers, no allocation, no
         *          exceptions.
         * @throws std::runtime_error for the same reasons @ref execute_reduction does.
         */
        template<template<typename> class Accum, typename Transform, typename Finalize>
        sycl::event execute_folding_reduction(Engine& engine, const Tensor& t, Tensor& result, const char* name, SushiRuntime::Graph::OpID op_id, Transform transform, Finalize finalize)
        {
            // Sized from the device the data lives on, which is the device the scheduler
            // routes the task to. Should the two ever disagree, Reduce::plan clamps to
            // whatever scratch actually exists and falls back to the single-pass shape -
            // slower, never wrong.
            const std::size_t partials =
                Reduce::planned_partials(Device::profile_for(t.get_device()),
                                         static_cast<std::size_t>(t.num_elements));

            auto dispatch = [transform, finalize]<typename T>(
                sycl::queue& q, const T* pT, T* pR, int64_t size,
                ReductionScratch<T> scratch, const std::vector<sycl::event>& deps)
            {
                return Reduce::reduce<Accum>(q, pT, pR, static_cast<std::size_t>(size),
                                             scratch.data, scratch.capacity,
                                             transform, finalize, deps);
            };

            return execute_reduction(engine, t, result, name, op_id, dispatch, partials);
        }

        /**
         * @brief The transform a reduction with nothing to transform passes.
         * @param x The element.
         * @return @p x.
         */
        struct ReduceIdentityTransform
        {
            /**
             * @brief Returns @p x unchanged.
             * @tparam T The element type.
             * @param x The element.
             * @return @p x.
             */
            template<typename T>
            [[nodiscard]] T operator()(T x) const noexcept { return x; }
        };

        /**
         * @brief The finalizer a reduction with nothing to finalize passes.
         */
        struct ReduceIdentityFinalize
        {
            /**
             * @brief Returns @p value unchanged.
             * @tparam T The element type.
             * @param value The folded result.
             * @param n The element count. Unused.
             * @return @p value.
             */
            template<typename T>
            [[nodiscard]] T operator()(T value, [[maybe_unused]] std::size_t n) const noexcept
            {
                return value;
            }
        };

    } // namespace Dispatch
} // namespace SushiBLAS
