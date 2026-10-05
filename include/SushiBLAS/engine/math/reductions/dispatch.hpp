/****************************************************************/
/* dispatch.hpp                                                 */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

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
#include <SushiBLAS/engine/reduce_axis.hpp>
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
         * @param dispatch A generic lambda with the signature `(sycl::queue& q,
         *        const auto* pT, auto* pR, int64_t size, auto scratch,
         *        const std::vector<sycl::event>& deps) -> sycl::event`, instantiated once
         *        per supported element type. `scratch` is a @ref ReductionScratch of the
         *        same element type as `pT`; it is spelled `auto` because a generic lambda
         *        deduces each parameter independently and so cannot name that type in its
         *        own parameter list.
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
         * @brief Records a whole-tensor reduction whose result is an **index**, not a value.
         *
         * `argmax` and `argmin` are not the same shape as `sum` or `nrm2` and were being
         * forced through @ref execute_reduction as though they were. That helper insists the
         * result tensor share the input's dtype and hands the dispatcher a `T*`, so an index
         * had to be stored as `static_cast<T>(i)` - a float. Below 2^24 elements that is
         * exact and nobody noticed; above it, two adjacent indices round to the same float
         * and the answer is quietly wrong. In HALF it is quietly wrong above 2048.
         *
         * Here the input dtype and the result dtype are separate questions: the input may be
         * anything an ordering is defined on, and the result is an `int64_t` in an INT64
         * tensor, which is what an index is.
         *
         * @tparam Dispatcher The command-group builder type, deduced.
         * @param engine The engine whose task graph the op is recorded on.
         * @param t The tensor to search.
         * @param result The scalar INT64 output tensor.
         * @param name Human-readable op name, used for logging and failure messages.
         * @param op_id Stable operation identifier for the task graph.
         * @param dispatch A generic callable `(sycl::queue& q, const auto* pT, int64_t* pR,
         *        int64_t size, const std::vector<sycl::event>& deps) -> sycl::event`,
         *        instantiated once per supported input element type.
         * @return An empty sycl::event; the work is deferred until `Engine::execute()`.
         *
         * @warning Same device-kernel constraints as @ref execute_reduction.
         * @throws std::runtime_error if `result` is not a scalar, if it is not INT64, or if
         *         the input dtype is one an ordering is not defined on (the complex types).
         */
        template<typename Dispatcher>
        sycl::event execute_index_reduction(Engine& engine, const Tensor& t, Tensor& result, const char* name, SushiRuntime::Graph::OpID op_id, Dispatcher&& dispatch)
        {
            SB_THROW_IF(result.num_elements != 1,
                        "Result tensor for reduction '{}' must be a scalar (1 element).", name);
            SB_THROW_IF(result.dtype != Core::DataType::INT64,
                        "Reduction '{}' answers with an index, so its result tensor must have "
                        "dtype INT64; got {}.",
                        name, Core::to_string(result.dtype));

            const int64_t size = t.num_elements;
            void* read_T = t.data();
            int64_t* write_R = result.data_as<int64_t>();

            std::vector<SushiRuntime::Core::ResourceRegion> reads = {t.region()};
            std::vector<SushiRuntime::Core::ResourceRegion> writes = {result.region()};

            SushiRuntime::Graph::TaskMetadata meta;
            meta.name = name;
            meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
            meta.op_id = op_id;

            // REAL_AND_INTEGRAL: "which element is largest" needs an ordering, which the
            // complex types do not have and the integer types do.
            with_element_type<Domain::REAL_AND_INTEGRAL>(t.dtype, "Reduction", name,
                [&](auto tag)
            {
                using T = typename decltype(tag)::type;

                engine.get_graph().add_task(meta, reads, writes,
                    [size, pT = static_cast<const T*>(read_T), write_R, dispatch, name]
                    (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
                    {
                        SB_LOG_INFO("Reduction {}: {} elements", name, size);
                        return dispatch(q, pT, write_R, size, deps);
                    });
            });

            return sycl::event();
        }

        /**
         * @brief Records a whole-tensor reduction that **counts** elements, answering into
         * an INT64 scalar.
         *
         * The third of this family's three executors, and the two it is not are the
         * clearest way to say what it is. @ref execute_reduction folds a tensor into a
         * scalar of the *same* dtype and owns a scratch buffer of that dtype;
         * @ref execute_index_reduction answers with an `int64_t` and owns no scratch,
         * because a serial scan for "which element" has no partials to stage. This one is
         * the remaining corner: an `int64_t` answer *and* a partials buffer, because the
         * question it serves - how many elements satisfy a predicate - is a fold over the
         * whole tensor and wants the two-stage shape rather than one work-item walking
         * every element.
         *
         * They stay three rather than one because merging them means making the result
         * element type a template parameter of a helper whose entire job is to hide the
         * element type, and every call site would then spell it. Three short functions
         * that each say what they answer with is the cheaper thing to read.
         *
         * **Why the count goes in an INT64 tensor.** It is exact for every tensor a
         * `Tensor` can describe, and it is a value a *device* kernel can consume with no
         * host round trip - which is the requirement, not a nicety: the caller is an
         * optimiser step that must decide whether to apply an update, and a host sync
         * there would serialise the training loop against the graph it is trying to keep
         * full. Storing it in the input's own dtype instead would reintroduce exactly the
         * defect @ref execute_index_reduction exists to have escaped - a count is only
         * exact in fp32 below 2^24 and in fp16 below 2048, and a HALF count past 65504
         * comes back as an infinity, so the tensors most worth counting are the ones whose
         * count would be wrong.
         *
         * @tparam Dispatcher The command-group builder type, deduced.
         * @param engine The engine whose task graph the op is recorded on.
         * @param t The tensor to scan.
         * @param result The scalar INT64 output tensor.
         * @param name Human-readable op name, used for logging and failure messages.
         * @param op_id Stable operation identifier for the task graph.
         * @param dispatch A generic callable `(sycl::queue& q, const auto* pT, int64_t* pR,
         *        int64_t size, ReductionScratch<int64_t> scratch,
         *        const std::vector<sycl::event>& deps) -> sycl::event`, instantiated once
         *        per supported input element type. `scratch` is spelled `auto` in a generic
         *        lambda for the reason @ref execute_reduction gives.
         * @param scratch_elements How many per-work-group partials the dispatcher wants
         *        room for; zero allocates nothing. @ref SushiBLAS::Reduce::planned_partials
         *        is how to compute it.
         * @return An empty sycl::event; the work is deferred until `Engine::execute()`.
         *
         * @warning Same device-kernel constraints as @ref execute_reduction.
         * @throws std::runtime_error if `result` is not a scalar, if it is not INT64, or if
         *         the input dtype is outside `Domain::REAL`.
         */
        template<typename Dispatcher>
        sycl::event execute_counting_reduction(Engine& engine, const Tensor& t, Tensor& result, const char* name, SushiRuntime::Graph::OpID op_id, Dispatcher&& dispatch, std::size_t scratch_elements = 0)
        {
            SB_THROW_IF(result.num_elements != 1,
                        "Result tensor for reduction '{}' must be a scalar (1 element).", name);
            SB_THROW_IF(result.dtype != Core::DataType::INT64,
                        "Reduction '{}' answers with a count, so its result tensor must have "
                        "dtype INT64; got {}. A count in a floating-point tensor is exact only "
                        "below 2^24 in FLOAT32 and below 2048 in HALF, and past 65504 a HALF "
                        "one is an infinity.",
                        name, Core::to_string(result.dtype));

            const int64_t size = t.num_elements;
            void* read_T = t.data();
            int64_t* write_R = result.data_as<int64_t>();

            // `Tensor::region()`, never the bare `data()` pointer - the reason is spelled
            // out at length in execute_reduction above, and it is the same reason here.
            std::vector<SushiRuntime::Core::ResourceRegion> reads = {t.region()};
            std::vector<SushiRuntime::Core::ResourceRegion> writes = {result.region()};

            // Partials are INT64 whatever the input dtype is, so the scratch is sized from
            // sizeof(int64_t) rather than from the tensor's element width.
            SushiRuntime::sushi_ptr<Storage> scratch;
            int64_t* scratch_ptr = nullptr;

            if (scratch_elements > 0)
            {
                const std::size_t bytes = scratch_elements * sizeof(int64_t);
                scratch = SushiRuntime::make_sushi<Storage>(
                    engine.get_context().get_allocator(), bytes,
                    SushiRuntime::Memory::Strategies::Device::instance());
                scratch_ptr = static_cast<int64_t*>(scratch->allocation_base());
                writes.push_back({scratch->allocation_base(), 0, bytes});
            }

            SushiRuntime::Graph::TaskMetadata meta;
            meta.name = name;
            meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
            meta.op_id = op_id;

            // REAL only. An integer or complex tensor is rejected by name rather than
            // answered with the zero that would technically be true of the integer case:
            // this is a screen for floating-point overflow, so a caller handing it a dtype
            // that cannot overflow has confused two tensors, and a confident "0 bad
            // elements" is the worst possible thing to tell them.
            with_element_type<Domain::REAL>(t.dtype, "Reduction", name,
                [&](auto tag)
            {
                using T = typename decltype(tag)::type;

                engine.get_graph().add_task(meta, reads, writes,
                    [size, pT = static_cast<const T*>(read_T), write_R, scratch,
                     pS = scratch_ptr, scratch_elements, dispatch, name]
                    (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
                    {
                        SB_LOG_INFO("Reduction {}: {} elements", name, size);
                        return dispatch(q, pT, write_R, size,
                                        ReductionScratch<int64_t>{pS, scratch_elements}, deps);
                    });
            });

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

            auto dispatch = [transform, finalize](
                sycl::queue& q, const auto* pT, auto* pR, int64_t size,
                auto scratch, const std::vector<sycl::event>& deps)
            {
                return Reduce::reduce<Accum>(q, pT, pR, static_cast<std::size_t>(size),
                                             scratch.data, scratch.capacity,
                                             transform, finalize, deps);
            };

            return execute_reduction(engine, t, result, name, op_id, dispatch, partials);
        }

        /**
         * @brief Resolves a rank-2 tensor and an axis into the stride pair a
         * segmented reduction walks, validating both operands on the way.
         *
         * The whole of "which axis, which layout, which view" lives here, on the
         * host, once. `Tensor::strides` already carries the `Core::Layout` the
         * tensor was built under and any non-unit stride a view inherited, so
         * reducing along either axis of either layout is the same arithmetic with
         * the two strides swapped - which is why @ref SushiBLAS::Reduce::AxisSpan
         * names no axis and the kernel carries no layout branch.
         *
         * Public rather than tucked into a detail namespace because it is the
         * reusable half of the seam: an index-valued axis reduction (`argmax`
         * along an axis) has a different *result dtype* from
         * @ref execute_folding_axis_reduction and so needs its own executor - the
         * way `execute_index_reduction` is a twin of `execute_reduction` - but it
         * has exactly this geometry, and should take it from here rather than
         * derive it a second time.
         *
         * @param t The tensor being reduced; must be rank 2.
         * @param axis Which axis is folded away: 0 reduces down the columns
         *        (result length `t.shape[1]`), 1 reduces along the rows (result
         *        length `t.shape[0]`).
         * @param result The rank-1 output, one element per surviving position.
         * @param name The op's own name, for the failure messages.
         * @return The span describing every segment.
         *
         * @throws std::runtime_error if @p t is not rank 2, if @p axis is not 0 or
         *         1, if @p result is not rank 1, or if @p result's length does not
         *         match the surviving extent.
         */
        [[nodiscard]] inline Reduce::AxisSpan axis_span_for(const Tensor& t, int axis,
                                                            const Tensor& result,
                                                            const char* name)
        {
            SB_THROW_IF(t.rank != 2,
                        "Axis reduction '{}' takes a rank-2 operand; got rank {}. Reducing one "
                        "axis of a higher-rank tensor is not implemented - the launch shape "
                        "generalises, the operand validation here does not.",
                        name, t.rank);
            SB_THROW_IF(axis != 0 && axis != 1,
                        "Axis reduction '{}' takes axis 0 or 1 on a rank-2 operand; got {}.",
                        name, axis);
            SB_THROW_IF(result.rank != 1,
                        "Axis reduction '{}' answers with a vector, so its result tensor must be "
                        "rank 1; got rank {}.",
                        name, result.rank);

            // The axis that survives is the one that is not folded away, and its
            // extent is how many independent folds there are.
            const int kept = 1 - axis;

            SB_THROW_IF(result.num_elements != t.shape[kept],
                        "Axis reduction '{}' over axis {} of a [{}, {}] tensor produces {} "
                        "values, but the result tensor holds {}.",
                        name, axis, t.shape[0], t.shape[1], t.shape[kept], result.num_elements);

            Reduce::AxisSpan span;
            span.segments       = static_cast<std::size_t>(t.shape[kept]);
            span.extent         = static_cast<std::size_t>(t.shape[axis]);
            span.segment_stride = static_cast<std::size_t>(t.strides[kept]);
            span.element_stride = static_cast<std::size_t>(t.strides[axis]);
            span.out_stride     = static_cast<std::size_t>(result.strides[0]);
            return span;
        }

        /**
         * @brief Resolves an NHWC tensor into the geometry that folds its two
         * middle axes away: `[N, H, W, C] -> [N, C]`.
         *
         * The spatial mean a normalisation layer needs, and the one shape
         * @ref axis_span_for's five numbers genuinely cannot describe. Folding two
         * axes at once is not the difficulty - `[N*H*W, C]` is a zero-copy rank-2
         * view and axis 0 of it folds to `[C]` with nothing new. The difficulty is
         * that the surviving position is a *pair* `(n, c)` whose input offset is
         * `n * strides[0] + c * strides[3]`, and no single `segment_stride` turns
         * one index into that. So @ref SushiBLAS::Reduce::AxisSpan grew a second
         * segment axis, and @ref axis_span_for's documented rank-2 restriction is
         * lifted exactly this far and no further: this is a named shape with its
         * own validation, not a general rank-N axis reduction.
         *
         * H and W are folded as a single run of `H * W` elements at stride
         * `strides[2]`, which is only the right answer when the rows are
         * contiguous with each other - hence the row-stride check, which rejects a
         * row-sliced view by name rather than reading past the end of one.
         *
         * @param t The tensor being reduced; must be rank 4, NHWC.
         * @param result The rank-2 `[N, C]` output, one element per surviving pair.
         * @param name The op's own name, for the failure messages.
         * @return The span describing every `(n, c)` fold.
         *
         * @throws std::runtime_error if @p t is not rank 4, if @p result is not
         *         rank 2, if @p result's shape is not `[N, C]`, or if @p t is a
         *         row-sliced view whose H and W cannot be walked as one run.
         */
        [[nodiscard]] inline Reduce::AxisSpan spatial_span_for(const Tensor& t, const Tensor& result,
                                                               const char* name)
        {
            SB_THROW_IF(t.rank != 4, "Reduction '{}' needs a rank-4 NHWC input; got rank {}.",
                        name, t.rank);
            SB_THROW_IF(result.rank != 2, "Reduction '{}' needs a rank-2 [N, C] result; got rank {}.",
                        name, result.rank);
            SB_THROW_IF(result.shape[0] != t.shape[0] || result.shape[1] != t.shape[3],
                        "Reduction '{}' folds [{}, {}, {}, {}] to [{}, {}] but was given a [{}, {}] result.",
                        name, t.shape[0], t.shape[1], t.shape[2], t.shape[3],
                        t.shape[0], t.shape[3], result.shape[0], result.shape[1]);
            SB_THROW_IF(t.strides[1] != t.strides[2] * t.shape[2],
                        "Reduction '{}' folds H and W as one run, which needs the row stride ({}) "
                        "to be the column stride ({}) times the width ({}). This tensor is a "
                        "row-sliced view; materialise it first.",
                        name, t.strides[1], t.strides[2], t.shape[2]);

            Reduce::AxisSpan span;
            span.segments  = static_cast<std::size_t>(t.shape[3]);              // c
            span.segments2 = static_cast<std::size_t>(t.shape[0]);              // n
            span.extent    = static_cast<std::size_t>(t.shape[1] * t.shape[2]); // h * w
            span.segment_stride  = static_cast<std::size_t>(t.strides[3]);
            span.segment_stride2 = static_cast<std::size_t>(t.strides[0]);
            span.element_stride  = static_cast<std::size_t>(t.strides[2]);
            span.out_stride      = static_cast<std::size_t>(result.strides[1]);
            return span;
        }

        /**
         * @brief Records a **segmented** reduction - one axis folded away, a vector
         * of results - using the library's own deterministic folding shape.
         *
         * The axis-wise counterpart of @ref execute_folding_reduction, and the seam
         * every axis reduction in the library goes through. `max_along` and
         * `sum_along` are each one call to this, differing only in @p Accum; a
         * per-axis mean is `Reduce::CompensatedSum` with `v / n` as its finalizer
         * and a per-axis minimum is a four-line accumulator, neither of which
         * re-cuts anything here.
         *
         * @code
         * // Row-wise maximum of a [rows, cols] tensor into a [rows] vector -
         * // the shift a numerically honest log-softmax needs.
         * Dispatch::execute_folding_axis_reduction<Reduce::RunningMax>(
         *     engine, x, 1, row_max, "math.max_along", "math.max_along"_op,
         *     Dispatch::ReduceIdentityTransform{}, Dispatch::ReduceIdentityFinalize{});
         * @endcode
         *
         * **It asks for no scratch**, unlike @ref execute_folding_reduction, and
         * that is a property of the shape rather than an omission: a segmented
         * reduction's work-groups never contribute to a shared answer, so there are
         * no partials to stage between two kernels. See
         * <SushiBLAS/engine/reduce_axis.hpp> for the two launch shapes and for the
         * determinism argument, which is the same one `reduce_1d.hpp` makes.
         *
         * @tparam Accum The accumulation policy: @ref SushiBLAS::Reduce::RunningMax,
         *         @ref SushiBLAS::Reduce::CompensatedSum for anything additive,
         *         @ref SushiBLAS::Reduce::RunningProduct for a product.
         * @tparam Transform The per-element transform type, deduced.
         * @tparam Finalize The finalizer type, deduced.
         * @param engine The engine whose task graph the op is recorded on.
         * @param t The tensor to reduce.
         * @param span The segment geometry, from @ref axis_span_for or
         *        @ref spatial_span_for. Nothing here re-derives it: this function
         *        takes the layout as given and only owns the graph bookkeeping,
         *        the dtype switch and the launch.
         * @param result The output tensor; must share `t`'s dtype.
         * @param name Human-readable op name, used for logging and failure messages.
         * @param op_id Stable operation identifier for the task graph.
         * @param transform Callable `(auto x) -> auto`, applied to each element
         *        before it is folded in.
         * @param finalize Callable `(auto value, std::size_t n) -> auto`, applied
         *        once per segment with @p n that segment's element count.
         * @return An empty sycl::event; the work is deferred until `Engine::execute()`.
         *
         * @warning @p transform and @p finalize are inlined into a SYCL device
         *          kernel and carry the same constraints as every other functor in
         *          this family: device-copyable, capture by value, no host
         *          pointers, no allocation, no exceptions.
         * @throws std::runtime_error if the dtypes disagree, or if the dtype is one reductions do not
         *         support (HALF, FLOAT32 and FLOAT64 are; the complex types have no
         *         ordering or identity to fold with, and an integer result is what
         *         the index reductions answer with, not this).
         */
        template<template<typename> class Accum, typename Transform, typename Finalize>
        sycl::event execute_folding_span_reduction(Engine& engine, const Tensor& t, const Reduce::AxisSpan& span, Tensor& result, const char* name, SushiRuntime::Graph::OpID op_id, Transform transform, Finalize finalize)
        {
            SB_THROW_IF(t.dtype != result.dtype,
                        "Data types must match for axis reduction '{}'.", name);

            void* read_T = t.data();
            void* write_R = result.data();

            // `Tensor::region()`, never the bare `data()` pointer - for exactly the
            // reason spelled out in execute_reduction above: a bare pointer widens
            // to the whole-buffer region of whatever it points at, so two views
            // inside one arena look like two different allocations, overlap() says
            // false, and the tracker emits no edge at all. That cost a training run
            // once already.
            std::vector<SushiRuntime::Core::ResourceRegion> reads = {t.region()};
            std::vector<SushiRuntime::Core::ResourceRegion> writes = {result.region()};

            SushiRuntime::Graph::TaskMetadata meta;
            meta.name = name;
            meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
            meta.op_id = op_id;

            // REAL only, matching execute_reduction: a fold needs an ordering and an
            // identity, which std::complex does not supply, and an integer tensor is
            // not an arithmetic type in this library (see Dispatch::Domain).
            with_element_type<Domain::REAL>(t.dtype, "Axis reduction", name,
                [&](auto tag)
            {
                using T = typename decltype(tag)::type;

                engine.get_graph().add_task(meta, reads, writes,
                    [span, pT = static_cast<const T*>(read_T), pR = static_cast<T*>(write_R),
                     transform, finalize, name]
                    (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
                    {
                        SB_LOG_INFO("Axis reduction {}: {} segments of {} elements",
                                    name, span.total_segments(), span.extent);

                        // Planned here, at execution time, from the device the task
                        // was actually routed to - the same reason Reduce::plan is
                        // asked inside the kernel launcher rather than at record
                        // time. Nothing is sized from it, so unlike the whole-tensor
                        // seam there is no scratch capacity to clamp against.
                        const Reduce::AxisPlan geometry =
                            Reduce::plan_axis(Device::profile_for(q), span.total_segments(),
                                              span.extent);

                        return Reduce::reduce_axis<Accum>(q, pT, pR, span, geometry,
                                                          transform, finalize, deps);
                    });
            });

            return sycl::event();
        }

        /**
         * @brief Records a segmented reduction over one axis of a rank-2 tensor.
         *
         * @ref execute_folding_span_reduction with @ref axis_span_for in front of
         * it, which is the split the spatial fold forced: the geometry a segmented
         * reduction runs on is no longer always derivable from "a tensor and an
         * axis number", but everything downstream of the span is identical, so the
         * span is the seam and the axis is one way of building one.
         *
         * @tparam Accum The accumulation policy; see @ref execute_folding_span_reduction.
         * @tparam Transform The per-element transform type, deduced.
         * @tparam Finalize The finalizer type, deduced.
         * @param engine The engine whose task graph the op is recorded on.
         * @param t The rank-2 tensor to reduce.
         * @param axis Which axis to fold away; see @ref axis_span_for.
         * @param result The rank-1 output tensor; must share `t`'s dtype.
         * @param name Human-readable op name, used for logging and failure messages.
         * @param op_id Stable operation identifier for the task graph.
         * @param transform Callable `(auto x) -> auto`, applied to each element
         *        before it is folded in.
         * @param finalize Callable `(auto value, std::size_t n) -> auto`, applied
         *        once per segment with @p n that segment's element count.
         * @return An empty sycl::event; the work is deferred until `Engine::execute()`.
         *
         * @warning Same device-kernel constraints as @ref execute_folding_span_reduction.
         * @throws std::runtime_error for the shape reasons @ref axis_span_for lists,
         *         or for the dtype reasons @ref execute_folding_span_reduction lists.
         */
        template<template<typename> class Accum, typename Transform, typename Finalize>
        sycl::event execute_folding_axis_reduction(Engine& engine, const Tensor& t, int axis, Tensor& result, const char* name, SushiRuntime::Graph::OpID op_id, Transform transform, Finalize finalize)
        {
            SB_THROW_IF(t.dtype != result.dtype,
                        "Data types must match for axis reduction '{}'.", name);

            const Reduce::AxisSpan span = axis_span_for(t, axis, result, name);

            return execute_folding_span_reduction<Accum>(engine, t, span, result, name, op_id,
                                                         transform, finalize);
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
