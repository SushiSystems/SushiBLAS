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
 * @brief Declares the reduction executors of the dispatch seam and the span builders they share.
 * @author Mustafa Garip
 */

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
#include <SushiBLAS/engine/scratch.hpp>
#include <SushiBLAS/support/reduce_1d.hpp>
#include <SushiBLAS/support/reduce_axis.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/support/device_policy.hpp>
#include <SushiRuntime/core/resource_region.hpp>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiBLAS
{
    namespace Dispatch
    {
        /**
         * @brief Holds a device buffer a reduction may stage per-work-group partials in.
         * @tparam T The partial type: Reduce::AccumulationOf the element type, or INT64 for a count.
         * @see include/SushiBLAS/engine/math/README.md
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
         * @brief Returns the bytes one partial of a floating-point reduction over dtype takes.
         * @return sizeof(Reduce::AccumulationOf<T>) for the element type T of dtype.
         */
        [[nodiscard]] inline std::size_t partial_bytes(Core::DataType dtype)
        {
            if (dtype == Core::DataType::HALF) return sizeof(Reduce::AccumulationOf<sycl::half>);
            return Core::bytes_per_element(dtype);
        }

        /**
         * @brief Returns the partials a whole-tensor reduction over t plans on t's own device.
         * @return The partial count, or 0 when t holds no elements.
         * @pre require_storage(t) has passed.
         * @see include/SushiBLAS/engine/math/README.md
         */
        [[nodiscard]] inline std::size_t planned_partials_for(const Tensor& t)
        {
            if (t.num_elements <= 0)
                return 0;

            return Reduce::planned_partials(Device::profile_for(t.get_device()),
                                            static_cast<std::size_t>(t.num_elements));
        }

        /**
         * @brief Records a whole-tensor reduction to a scalar result tensor.
         * @param result Must hold one element and share t's dtype.
         * @param dispatch Builds the command group per element type and must call `depends_on(deps)`.
         * @param scratch_elements Per-work-group partials to allocate; zero allocates nothing.
         * @throws std::invalid_argument if an operand holds elements without storage or t is not contiguous.
         * @throws std::runtime_error if result is not scalar, the dtypes disagree, or the dtype is not real.
         * @see include/SushiBLAS/engine/math/README.md
         */
        template<typename Dispatcher>
        sycl::event execute_reduction(Engine& engine, const Tensor& t, Tensor& result, const char* name, SushiRuntime::Graph::OpID op_id, Dispatcher&& dispatch, std::size_t scratch_elements = 0)
        {
            require_storage(t, name, "t");
            require_storage(result, name, "result");
            require_contiguous(t, name, "t");
            SB_THROW_IF(result.num_elements != 1, "Result tensor for reduction '{}' must be a scalar (1 element).", name);
            SB_THROW_IF(t.dtype != result.dtype, "Data types must match for reduction operations.");

            int64_t size = t.num_elements;
            const void* read_T = t.data();
            void* write_R = result.data();

            // Operands are declared by Tensor::region(), not by data(); see include/SushiBLAS/engine/math/README.md.
            std::vector<SushiRuntime::Core::ResourceRegion> reads = {t.region()};
            std::vector<SushiRuntime::Core::ResourceRegion> writes = {result.region()};

            // The scratch is declared as a resource with its own region; see include/SushiBLAS/engine/math/README.md.
            const ScratchBuffer scratch(engine, scratch_elements * partial_bytes(t.dtype),
                                        t.storage->get_device_index());
            void* scratch_ptr = scratch.data();
            if (!scratch.is_empty()) writes.push_back(scratch.region());

            SushiRuntime::Graph::TaskMetadata meta;
            meta.name = name;
            meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
            meta.op_id = op_id;

            switch (t.dtype)
            {
                case Core::DataType::HALF:
                    engine.get_graph().add_task(meta, reads, writes,
                        [size, pT=(const sycl::half*)read_T, pR=(sycl::half*)write_R, scratch,
                         pS=(float*)scratch_ptr, scratch_elements, dispatch, name]
                        (sycl::queue& q, const std::vector<sycl::event>& deps) -> sycl::event
                        {
                            SB_LOG_INFO("Reduction {}: {} elements", name, size);
                            return dispatch(q, pT, pR, size,
                                            ReductionScratch<float>{pS, scratch_elements}, deps);
                        });
                    break;
                case Core::DataType::FLOAT32:
                    engine.get_graph().add_task(meta, reads, writes,
                        [size, pT=(const float*)read_T, pR=(float*)write_R, scratch,
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
                        [size, pT=(const double*)read_T, pR=(double*)write_R, scratch,
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
         * @brief Records a whole-tensor reduction whose result is an index in an INT64 scalar.
         * @param result Must be a scalar INT64 tensor.
         * @param dispatch Builds the command group per input element type, as in execute_reduction.
         * @throws std::invalid_argument if an operand holds elements without storage or t is not contiguous.
         * @throws std::runtime_error if result is not an INT64 scalar or the input dtype is complex.
         * @see include/SushiBLAS/engine/math/README.md
         */
        template<typename Dispatcher>
        sycl::event execute_index_reduction(Engine& engine, const Tensor& t, Tensor& result, const char* name, SushiRuntime::Graph::OpID op_id, Dispatcher&& dispatch)
        {
            require_storage(t, name, "t");
            require_storage(result, name, "result");
            require_contiguous(t, name, "t");
            SB_THROW_IF(result.num_elements != 1,
                        "Result tensor for reduction '{}' must be a scalar (1 element).", name);
            SB_THROW_IF(result.dtype != Core::DataType::INT64,
                        "Reduction '{}' answers with an index, so its result tensor must have "
                        "dtype INT64; got {}.",
                        name, Core::to_string(result.dtype));

            const int64_t size = t.num_elements;
            const void* read_T = t.data();
            int64_t* write_R = result.data_as<int64_t>();

            std::vector<SushiRuntime::Core::ResourceRegion> reads = {t.region()};
            std::vector<SushiRuntime::Core::ResourceRegion> writes = {result.region()};

            SushiRuntime::Graph::TaskMetadata meta;
            meta.name = name;
            meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
            meta.op_id = op_id;

            // REAL_AND_INTEGRAL: finding the largest element needs an ordering, which complex types lack; see include/SushiBLAS/engine/math/README.md.
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
         * @brief Records a whole-tensor reduction that counts elements into an INT64 scalar.
         * @param result Must be a scalar INT64 tensor.
         * @param scratch_elements Per-work-group partials to allocate; zero allocates nothing.
         * @throws std::invalid_argument if an operand holds elements without storage or t is not contiguous.
         * @throws std::runtime_error if result is not an INT64 scalar or the input dtype is not real.
         * @see include/SushiBLAS/engine/math/README.md
         */
        template<typename Dispatcher>
        sycl::event execute_counting_reduction(Engine& engine, const Tensor& t, Tensor& result, const char* name, SushiRuntime::Graph::OpID op_id, Dispatcher&& dispatch, std::size_t scratch_elements = 0)
        {
            require_storage(t, name, "t");
            require_storage(result, name, "result");
            require_contiguous(t, name, "t");
            SB_THROW_IF(result.num_elements != 1,
                        "Result tensor for reduction '{}' must be a scalar (1 element).", name);
            SB_THROW_IF(result.dtype != Core::DataType::INT64,
                        "Reduction '{}' answers with a count, so its result tensor must have "
                        "dtype INT64; got {}. A count in a floating-point tensor is exact only "
                        "below 2^24 in FLOAT32 and below 2048 in HALF, and past 65504 a HALF "
                        "one is an infinity.",
                        name, Core::to_string(result.dtype));

            const int64_t size = t.num_elements;
            const void* read_T = t.data();
            int64_t* write_R = result.data_as<int64_t>();

            // Operands are declared by Tensor::region(), not by data(); see include/SushiBLAS/engine/math/README.md.
            std::vector<SushiRuntime::Core::ResourceRegion> reads = {t.region()};
            std::vector<SushiRuntime::Core::ResourceRegion> writes = {result.region()};

            // Partials are INT64 whatever the input dtype, so the scratch is sized from sizeof(int64_t); see include/SushiBLAS/engine/math/README.md.
            const ScratchBuffer scratch(engine, scratch_elements * sizeof(int64_t),
                                        t.storage->get_device_index());
            int64_t* scratch_ptr = static_cast<int64_t*>(scratch.data());
            if (!scratch.is_empty()) writes.push_back(scratch.region());

            SushiRuntime::Graph::TaskMetadata meta;
            meta.name = name;
            meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
            meta.op_id = op_id;

            // REAL only: an integer or complex tensor is rejected by name; see include/SushiBLAS/engine/math/README.md.
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
         * @brief Records a whole-tensor reduction as `result = finalize(fold(transform(t[i])), n)`.
         * @tparam Accum Reduce::CompensatedSum for anything additive, Reduce::RunningProduct for a product.
         * @param transform Callable `(auto x) -> auto`, applied to each element before the fold.
         * @param finalize Callable `(auto value, std::size_t n) -> auto`, applied once to the folded result.
         * @throws std::invalid_argument for the same reasons execute_reduction does.
         * @throws std::runtime_error for the same reasons execute_reduction does.
         * @see include/SushiBLAS/engine/math/README.md
         */
        template<template<typename> class Accum, typename Transform, typename Finalize>
        sycl::event execute_folding_reduction(Engine& engine, const Tensor& t, Tensor& result, const char* name, SushiRuntime::Graph::OpID op_id, Transform transform, Finalize finalize)
        {
            require_storage(t, name, "t");
            require_storage(result, name, "result");

            const std::size_t partials = planned_partials_for(t);

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
         * @brief Resolves a rank-2 tensor and an axis into the stride pair a segmented reduction walks.
         * @param axis 0 reduces down the columns, 1 along the rows.
         * @param result Rank-1 tensor with one element per surviving position.
         * @throws std::runtime_error on a rank, axis or length mismatch.
         * @see include/SushiBLAS/engine/math/README.md
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

            // The surviving axis is the one not folded away; its extent is the number of folds; see include/SushiBLAS/engine/math/README.md.
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
         * @brief Resolves an NHWC tensor into the geometry that folds H and W away: `[N, H, W, C] -> [N, C]`.
         * @param t Must be rank 4, NHWC, with rows contiguous with one another.
         * @param result Rank-2 `[N, C]` tensor with one element per surviving pair.
         * @throws std::runtime_error on a rank or shape mismatch, or a row-sliced view.
         * @see include/SushiBLAS/engine/math/README.md
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
         * @brief Records a segmented reduction over a span, one result per segment.
         * @param span The segment geometry, from axis_span_for or spatial_span_for.
         * @param result Must share t's dtype.
         * @param finalize Callable `(auto value, std::size_t n) -> auto`, applied once per segment.
         * @throws std::invalid_argument if an operand holds elements without storage.
         * @throws std::runtime_error if the dtypes disagree or the dtype is not HALF, FLOAT32 or FLOAT64.
         * @see include/SushiBLAS/engine/math/README.md
         */
        template<template<typename> class Accum, typename Transform, typename Finalize>
        sycl::event execute_folding_span_reduction(Engine& engine, const Tensor& t, const Reduce::AxisSpan& span, Tensor& result, const char* name, SushiRuntime::Graph::OpID op_id, Transform transform, Finalize finalize)
        {
            require_storage(t, name, "t");
            require_storage(result, name, "result");
            SB_THROW_IF(t.dtype != result.dtype,
                        "Data types must match for axis reduction '{}'.", name);

            const void* read_T = t.data();
            void* write_R = result.data();

            // Operands are declared by Tensor::region(), not by data(); see include/SushiBLAS/engine/math/README.md.
            std::vector<SushiRuntime::Core::ResourceRegion> reads = {t.region()};
            std::vector<SushiRuntime::Core::ResourceRegion> writes = {result.region()};

            SushiRuntime::Graph::TaskMetadata meta;
            meta.name = name;
            meta.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
            meta.op_id = op_id;

            // REAL only: a fold needs an ordering and an identity, and integers are not arithmetic here; see include/SushiBLAS/engine/math/README.md.
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

                        // The geometry is planned at execution time; see include/SushiBLAS/engine/math/README.md.
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
         * @param axis Which axis to fold away; see axis_span_for.
         * @param result Rank-1 tensor sharing t's dtype.
         * @throws std::invalid_argument if an operand holds elements without storage.
         * @throws std::runtime_error for the reasons axis_span_for and execute_folding_span_reduction list.
         * @see include/SushiBLAS/engine/math/README.md
         */
        template<template<typename> class Accum, typename Transform, typename Finalize>
        sycl::event execute_folding_axis_reduction(Engine& engine, const Tensor& t, int axis, Tensor& result, const char* name, SushiRuntime::Graph::OpID op_id, Transform transform, Finalize finalize)
        {
            require_storage(t, name, "t");
            require_storage(result, name, "result");
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
