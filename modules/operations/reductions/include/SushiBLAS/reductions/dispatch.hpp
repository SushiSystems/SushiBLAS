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
#include <utility>
#include <type_traits>
#include <sycl/sycl.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/tensor/storage.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/reductions/reductions_log_category.hpp>
#include <SushiBLAS/engine/dispatch.hpp>
#include <SushiBLAS/engine/scratch.hpp>
#include <SushiBLAS/reduction/reduce_1d.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/task_handle.hpp>
#include <SushiBLAS/reduction/reduce_axis.hpp>
#include <SushiBLAS/engine/operand_check.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiBLAS/launch/device_policy.hpp>
#include <SushiBLAS/reference/serial_reduce.hpp>
#include <SushiBLAS/engine/execution_mode.hpp>
#include <SushiBLAS/engine/kernel_pair_dispatcher.hpp>
#include <SushiBLAS/engine/task_description.hpp>
#include <SushiRuntime/core/operation_id.hpp>

namespace SushiBLAS
{
    namespace Dispatch
    {
        /**
         * @brief Holds a device buffer a reduction may stage per-work-group partials in.
         * @tparam T The partial type: Reduce::AccumulationOf the element type, or INT64 for a count.
         * @see modules/operations/MATH_README.md
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
            if (dtype == Core::DataType::Half)
                return sizeof(Reduce::AccumulationOf<sycl::half>);
            return Core::bytes_per_element(dtype);
        }

        /**
         * @brief Returns the partials a whole-tensor reduction over t plans on t's own device.
         * @return The partial count, or 0 when t holds no elements.
         * @pre require_storage(t) has passed.
         * @see modules/operations/MATH_README.md
         */
        [[nodiscard]] inline std::size_t planned_partials_for(const Tensor& t)
        {
            if (t.num_elements <= 0)
                return 0;

            return Reduce::planned_partials(Device::profile_for(t.device()),
                                            static_cast<std::size_t>(t.num_elements));
        }

        /**
         * @brief Returns the device queue a reduction of t into result leases its scratch on.
         * @pre result has storage.
         */
        [[nodiscard]] inline std::size_t reduction_device_index(const Tensor& t,
                                                                const Tensor& result)
        {
            return t.storage ? t.storage->device_index()
                             : result.storage->device_index();
        }

        /** @brief Holds the bytes one (value, index) partial of an index reduction takes. */
        inline constexpr std::size_t K_INDEXED_PARTIAL_BYTES = 16;

        /**
         * @brief Returns how many partials a reduction leases under the recorder's mode.
         * @return planned, or 0 under ForceReference, whose host fold stages nothing.
         */
        [[nodiscard]] inline std::size_t leased_partials(const TaskRecorder& recorder,
                                                         std::size_t planned) noexcept
        {
            return recorder.mode() == ExecutionMode::ForceReference ? std::size_t{0}
                                                                          : planned;
        }

        namespace Detail
        {
            /**
             * @brief Reports whether Dispatcher has the host path, `dispatch.reference(in, out, size)`.
             * @see modules/operations/MATH_README.md
             */
            template<typename Dispatcher, typename T, typename Out, typename = void>
            struct has_reference_path : std::false_type {};

            template<typename Dispatcher, typename T, typename Out>
            struct has_reference_path<Dispatcher, T, Out,
                                      std::void_t<decltype(std::declval<const Dispatcher&>().reference(
                                          std::declval<const T*>(), std::declval<Out*>(),
                                          std::declval<int64_t>()))>>
                : std::true_type {};

            /** @brief Reports whether R is the C++ type of HALF, FLOAT32 or FLOAT64. */
            template<typename R>
            inline constexpr bool is_real_scalar_v =
                std::is_floating_point_v<R> || std::is_same_v<R, sycl::half>;

            /**
             * @brief Reports whether Finalize, handed a fold in Wide, returns a value no narrower than Wide.
             * @tparam Wide The type a host fold hands finalize: double for a sum, whatever the element type.
             * @see modules/operations/MATH_README.md
             */
            template<typename Finalize, typename Wide>
            struct finalize_keeps_width
            {
                /** @brief The type finalize returns for a fold in Wide. */
                using Result = std::decay_t<std::invoke_result_t<const Finalize&, Wide, std::size_t>>;

                /** @brief Holds false only when both types are real scalars and Result is the narrower. */
                static constexpr bool value = !is_real_scalar_v<Result> || !is_real_scalar_v<Wide> ||
                                              sizeof(Result) >= sizeof(Wide);
            };

            /**
             * @brief Refuses at compile time a Finalize that narrows the host fold it is handed.
             * @tparam Wide The type a host fold hands finalize: double for a sum, whatever the element type.
             * @see modules/operations/MATH_README.md
             */
            template<typename Finalize, typename Wide>
            constexpr void require_finalize_keeps_width() noexcept
            {
                // The static_assert states the finalize contract in a sentence; see modules/operations/MATH_README.md.
                static_assert(finalize_keeps_width<Finalize, Wide>::value,
                              "A reduction's finalize must return its value in the type it is "
                              "handed, which is double on the host path of a sum whatever the "
                              "element type. Write it as '[](auto value, std::size_t n) { ... }' "
                              "and let the seam narrow the result; this one returns a narrower "
                              "type.");
            }

            /**
             * @brief Runs one whole-tensor reduction on the host or the device, as its mode says.
             * @tparam P The type of one staged partial.
             * @tparam Out The type the result is written in.
             * @tparam Dispatcher Callable as `dispatch(q, in, out, size, ReductionScratch<P>, deps)`
             *         for the SYCL path, with a member `reference(in, out, size)` for the host path.
             * @see modules/operations/MATH_README.md
             */
            template<typename T, typename P, typename Out, typename Dispatcher>
            class ReductionTask final
            {
                /// @brief What the dispatcher's SYCL path returns: the event of its kernel, or the events of all of them.
                using Result = std::decay_t<std::invoke_result_t<
                    const Dispatcher&, sycl::queue&, const T*, Out*, int64_t,
                    ReductionScratch<P>, const std::vector<sycl::event>&>>;

                // The static_assert states the dispatcher contract in a sentence; see modules/operations/MATH_README.md.
                static_assert(has_reference_path<Dispatcher, T, Out>::value,
                              "A reduction dispatcher needs both paths: "
                              "'operator()(sycl::queue&, const T*, Out*, int64_t, "
                              "ReductionScratch<P>, const std::vector<sycl::event>&) const', "
                              "returning a sycl::event or Device::SubmittedEvents, for "
                              "the SYCL fold and 'void reference(const T*, Out*, int64_t) const' "
                              "for the serial host fold. This one has no reference(); a lambda "
                              "cannot be a dispatcher, write a class.");

                public:
                    /**
                     * @brief Holds everything the body needs at execute time.
                     * @param scratch The lease on the partials; held so the block stays lent to this task.
                     */
                    ReductionTask(const T* in, Out* out, int64_t size, ScratchBuffer scratch,
                                  std::size_t partials, Dispatcher dispatch)
                        : in_(in), out_(out), size_(size), scratch_(std::move(scratch)),
                          partials_(partials), dispatch_(std::move(dispatch))
                    {

                    }

                    /** @brief Runs the serial host fold under ForceReference, else submits the kernels. */
                    Result operator()(const KernelPairDispatcher& pair,
                                      sycl::queue& q,
                                      const std::vector<sycl::event>& deps) const
                    {

                        return pair.run(q, deps,
                            [this] { dispatch_.reference(in_, out_, size_); },
                            [this](sycl::queue& q2,
                                   const std::vector<sycl::event>& deps2) -> Result
                            {
                                return dispatch_(q2, in_, out_, size_,
                                                 ReductionScratch<P>{
                                                     static_cast<P*>(scratch_.data()), partials_},
                                                 deps2);
                            });
                    }

                private:
                    /// @brief The input's base address; null for an operand without storage.
                    const T* in_;

                    /// @brief The address the result is written to.
                    Out* out_;

                    /// @brief The element count of the input.
                    int64_t size_;

                    /// @brief The lease on the staged partials; empty under ForceReference.
                    ScratchBuffer scratch_;

                    /// @brief How many partials the lease has room for.
                    std::size_t partials_;

                    /// @brief The reduction's two paths.
                    Dispatcher dispatch_;
            };
        } // namespace Detail

        /**
         * @brief Records a whole-tensor reduction to a scalar result tensor.
         * @param result Must hold one element and share t's dtype.
         * @param dispatch Has the two paths Detail::ReductionTask names; its SYCL path must call `depends_on(deps)`.
         * @param scratch_elements Per-work-group partials to lease; zero leases nothing.
         * @throws std::invalid_argument on an operand violation; UnsupportedExecution on a device-only tensor under ForceReference or a dtype a tensor's device cannot compute in.
         * @see modules/operations/MATH_README.md
         */
        template<typename Dispatcher>
        TaskHandle execute_reduction(TaskRecorder& recorder, const Tensor& t, Tensor& result,
                                     const char* name, SushiRuntime::OperationID op_id,
                                     Dispatcher&& dispatch, std::size_t scratch_elements = 0)
        {
            require_storage(t, name, "t");
            require_device_dtype(t, name, "t");
            require_storage(result, name, "result");
            require_device_dtype(result, name, "result");
            require_contiguous(t, name, "t");
            require_scalar(result, name, "result");
            require_dtype(result, t.dtype, name, "result");
            require_domain<Domain::Real>(t.dtype, "Reduction", name);

            int64_t size = t.num_elements;
            const void* read_T = t.data_or_null();
            void* write_R = result.data();

            // Operands are declared by Tensor::region(), not by data(); see modules/operations/MATH_README.md.
            OperandSet operands;
            operands.reads(t).writes(result);
            recorder.require_host_reachable(operands, "Reduction", name);

            // The scratch is declared as a resource with its own region; see modules/operations/MATH_README.md.
            const std::size_t partials = leased_partials(recorder, scratch_elements);
            const ScratchBuffer scratch(recorder.scratch_pool(),
                                        partials * partial_bytes(t.dtype),
                                        reduction_device_index(t, result));
            operands.scratch(scratch);

            TaskDescription description;
            description.name = name;
            description.operation = op_id;
            description.shape = ReductionShape{size, 1, t.dtype};

            using D = std::decay_t<Dispatcher>;

            switch (t.dtype)
            {
                case Core::DataType::Half:
                    return recorder.record_pair(description, operands, "Reduction",
                        Detail::ReductionTask<sycl::half, float, sycl::half, D>(
                            static_cast<const sycl::half*>(read_T),
                            static_cast<sycl::half*>(write_R), size, scratch, partials, dispatch));
                case Core::DataType::Float32:
                    return recorder.record_pair(description, operands, "Reduction",
                        Detail::ReductionTask<float, float, float, D>(
                            static_cast<const float*>(read_T), static_cast<float*>(write_R), size,
                            scratch, partials, dispatch));
                case Core::DataType::Float64:
                    return recorder.record_pair(description, operands, "Reduction",
                        Detail::ReductionTask<double, double, double, D>(
                            static_cast<const double*>(read_T), static_cast<double*>(write_R),
                            size, scratch, partials, dispatch));
                default:
                    break;
            }

            SB_THROW_IF(true, "Reduction '{}' does not support dtype {}.", name, Core::to_string(t.dtype));
            return TaskHandle();
        }

        namespace Detail
        {
            /**
             * @brief Records an index reduction for the element type of a with_element_type arm.
             * @tparam Dispatcher The two-path dispatcher execute_index_reduction was given.
             * @see docs/backlog/README.md
             */
            template<typename Dispatcher>
            struct IndexReductionArm
            {
                /** @brief Holds the recorder the task is recorded through. */
                TaskRecorder& recorder;
                /** @brief Holds the metadata the task is recorded with. */
                const TaskDescription& description;
                /** @brief Holds the operands the task declares. */
                const OperandSet& operands;
                /** @brief Holds the dispatcher the task body calls. */
                const Dispatcher& dispatch;
                /** @brief Holds the element count of the input. */
                int64_t size;
                /** @brief Holds the input's base address. */
                const void* read_T;
                /** @brief Holds the address the index is written to. */
                int64_t* write_R;
                /** @brief Holds the lease the task keeps until it is destroyed. */
                const ScratchBuffer& scratch;
                /** @brief Holds how many partials the lease has room for. */
                std::size_t partials;
                /** @brief Receives the handle of the recorded task. */
                TaskHandle& handle;

                /** @brief Records the task for the element type named by the tag. */
                template<typename Tag>
                void operator()(Tag) const
                {
                    using T = typename Tag::type;
                    using P = Reduce::IndexedValue<T>;

                    static_assert(sizeof(P) <= K_INDEXED_PARTIAL_BYTES,
                                  "An index partial must fit the bytes its scratch is sized by.");

                    handle = recorder.record_pair(description, operands, "Reduction",
                        ReductionTask<T, P, int64_t, Dispatcher>(
                            static_cast<const T*>(read_T), write_R, size, scratch, partials,
                            dispatch));
                }
            };

            /**
             * @brief Records a counting reduction for the element type of a with_element_type arm.
             * @tparam Dispatcher The two-path dispatcher execute_counting_reduction was given.
             * @see docs/backlog/README.md
             */
            template<typename Dispatcher>
            struct CountingReductionArm
            {
                /** @brief Holds the recorder the task is recorded through. */
                TaskRecorder& recorder;
                /** @brief Holds the metadata the task is recorded with. */
                const TaskDescription& description;
                /** @brief Holds the operands the task declares. */
                const OperandSet& operands;
                /** @brief Holds the dispatcher the task body calls. */
                const Dispatcher& dispatch;
                /** @brief Holds the element count of the input. */
                int64_t size;
                /** @brief Holds the input's base address. */
                const void* read_T;
                /** @brief Holds the address the count is written to. */
                int64_t* write_R;
                /** @brief Holds the lease the task keeps until it is destroyed. */
                const ScratchBuffer& scratch;
                /** @brief Holds how many partials the lease has room for. */
                std::size_t partials;
                /** @brief Receives the handle of the recorded task. */
                TaskHandle& handle;

                /** @brief Records the task for the element type named by the tag. */
                template<typename Tag>
                void operator()(Tag) const
                {
                    using T = typename Tag::type;

                    handle = recorder.record_pair(description, operands, "Reduction",
                        ReductionTask<T, int64_t, int64_t, Dispatcher>(
                            static_cast<const T*>(read_T), write_R, size, scratch, partials,
                            dispatch));
                }
            };

            /**
             * @brief Runs one segmented fold on the host or the device, as its mode says.
             * @tparam Accum The Reduce accumulator the fold runs in.
             * @tparam Finalize Returns its value in the type it is handed: double on the host path of a sum.
             * @see modules/operations/MATH_README.md
             */
            template<template<typename> class Accum, typename T, typename Transform,
                     typename Finalize>
            class SpanReductionTask final
            {
                public:
                    /**
                     * @brief Holds everything the body needs at execute time.
                     */
                    SpanReductionTask(const Reduce::AxisSpan& span, const T* in, T* out,
                                      Transform transform, Finalize finalize)
                        : span_(span), in_(in), out_(out), transform_(std::move(transform)),
                          finalize_(std::move(finalize))
                    {

                    }

                    /** @brief Runs the serial host folds under ForceReference, else submits the kernel. */
                    sycl::event operator()(const KernelPairDispatcher& pair,
                                           sycl::queue& q,
                                           const std::vector<sycl::event>& deps) const
                    {

                        return pair.run(q, deps,
                            [this]
                            {
                                require_finalize_keeps_width<
                                    Finalize, typename Serial::Fold<Accum, T>::Wide>();

                                Serial::reduce_axis<Accum>(in_, out_, span_, transform_, finalize_);
                            },
                            [this](sycl::queue& q2,
                                   const std::vector<sycl::event>& deps2) -> sycl::event
                            {
                                // The geometry is planned at execution time; see modules/operations/MATH_README.md.
                                const Reduce::AxisPlan geometry =
                                    Reduce::plan_axis(Device::profile_for(q2),
                                                      span_.total_segments(), span_.extent);

                                return Reduce::reduce_axis<Accum>(q2, in_, out_, span_, geometry,
                                                                  transform_, finalize_, deps2);
                            });
                    }

                private:
                    /// @brief The segment geometry the fold walks.
                    Reduce::AxisSpan span_;

                    /// @brief The input's base address; null for an operand without storage.
                    const T* in_;

                    /// @brief The result's base address.
                    T* out_;

                    /// @brief The callable applied to each element before the fold.
                    Transform transform_;

                    /// @brief The callable applied once to each segment's fold.
                    Finalize finalize_;
            };

            /**
             * @brief Records a segmented fold for the element type of a with_element_type arm.
             * @tparam Accum The Reduce accumulator the fold runs in.
             * @see docs/backlog/README.md
             */
            template<template<typename> class Accum, typename Transform, typename Finalize>
            struct SpanReductionArm
            {
                /** @brief Holds the recorder the task is recorded through. */
                TaskRecorder& recorder;
                /** @brief Holds the metadata the task is recorded with. */
                const TaskDescription& description;
                /** @brief Holds the operands the task declares. */
                const OperandSet& operands;
                /** @brief Holds the segment geometry the fold walks. */
                const Reduce::AxisSpan& span;
                /** @brief Holds the input's base address. */
                const void* read_T;
                /** @brief Holds the result's base address. */
                void* write_R;
                /** @brief Holds the callable applied to each element before the fold. */
                const Transform& transform;
                /** @brief Holds the callable applied once to each segment's fold. */
                const Finalize& finalize;
                /** @brief Receives the handle of the recorded task. */
                TaskHandle& handle;

                /** @brief Records the task for the element type named by the tag. */
                template<typename Tag>
                void operator()(Tag) const
                {
                    using T = typename Tag::type;

                    handle = recorder.record_pair(description, operands, "Axis reduction",
                        SpanReductionTask<Accum, T, Transform, Finalize>(
                            span, static_cast<const T*>(read_T), static_cast<T*>(write_R),
                            transform, finalize));
                }
            };
        } // namespace Detail

        /**
         * @brief Records a whole-tensor reduction whose result is an index in an INT64 scalar.
         * @param result Must be a scalar INT64 tensor.
         * @param dispatch Has the two paths Detail::ReductionTask names, over IndexedValue partials.
         * @param scratch_elements Per-work-group partials to lease; zero leases nothing.
         * @throws std::invalid_argument on an operand violation or an empty t; UnsupportedExecution on a device-only tensor under ForceReference or a dtype a tensor's device cannot compute in.
         * @see modules/operations/MATH_README.md
         */
        template<typename Dispatcher>
        TaskHandle execute_index_reduction(TaskRecorder& recorder, const Tensor& t, Tensor& result,
                                           const char* name, SushiRuntime::OperationID op_id,
                                           Dispatcher&& dispatch,
                                           std::size_t scratch_elements = 0)
        {
            require_storage(t, name, "t");
            require_device_dtype(t, name, "t");
            require_storage(result, name, "result");
            require_device_dtype(result, name, "result");
            require_contiguous(t, name, "t");
            require_scalar(result, name, "result");
            require_dtype(result, Core::DataType::Int64, name, "result");
            require_domain<Domain::RealAndIntegral>(t.dtype, "Reduction", name);
            require_min_elements(t, 1, name, "t");

            const int64_t size = t.num_elements;
            const void* read_T = t.data();
            int64_t* write_R = result.data_as<int64_t>();

            OperandSet operands;
            operands.reads(t).writes(result);
            recorder.require_host_reachable(operands, "Reduction", name);

            // The partials are (value, index) pairs, 16 bytes for every element type served.
            const std::size_t partials = leased_partials(recorder, scratch_elements);
            const ScratchBuffer scratch(recorder.scratch_pool(),
                                        partials * K_INDEXED_PARTIAL_BYTES,
                                        t.storage->device_index());
            operands.scratch(scratch);

            TaskDescription description;
            description.name = name;
            description.operation = op_id;
            description.shape = ReductionShape{size, 1, t.dtype};

            TaskHandle handle;

            // REAL_AND_INTEGRAL: finding the largest element needs an ordering, which complex types lack; see modules/operations/MATH_README.md.
            const Detail::IndexReductionArm<std::remove_reference_t<Dispatcher>> arm{
                recorder, description, operands, dispatch, size, read_T, write_R, scratch, partials,
                handle};
            with_element_type<Domain::RealAndIntegral>(t.dtype, "Reduction", name, arm);

            return handle;
        }

        /**
         * @brief Records a whole-tensor reduction that counts elements into an INT64 scalar.
         * @param result Must be a scalar INT64 tensor.
         * @param dispatch Has the two paths Detail::ReductionTask names, over INT64 partials.
         * @param scratch_elements Per-work-group partials to lease; zero leases nothing.
         * @throws std::invalid_argument on an operand violation; UnsupportedExecution on a device-only tensor under ForceReference or a dtype a tensor's device cannot compute in.
         * @see modules/operations/MATH_README.md
         */
        template<typename Dispatcher>
        TaskHandle execute_counting_reduction(TaskRecorder& recorder, const Tensor& t,
                                              Tensor& result, const char* name,
                                              SushiRuntime::OperationID op_id,
                                              Dispatcher&& dispatch,
                                              std::size_t scratch_elements = 0)
        {
            require_storage(t, name, "t");
            require_device_dtype(t, name, "t");
            require_storage(result, name, "result");
            require_device_dtype(result, name, "result");
            require_contiguous(t, name, "t");
            require_scalar(result, name, "result");
            require_dtype(result, Core::DataType::Int64, name, "result");
            require_domain<Domain::Real>(t.dtype, "Reduction", name);

            const int64_t size = t.num_elements;
            const void* read_T = t.data_or_null();
            int64_t* write_R = result.data_as<int64_t>();

            // Operands are declared by Tensor::region(), not by data(); see modules/operations/MATH_README.md.
            OperandSet operands;
            operands.reads(t).writes(result);
            recorder.require_host_reachable(operands, "Reduction", name);

            // Partials are INT64 whatever the input dtype, so the scratch is sized from sizeof(int64_t); see modules/operations/MATH_README.md.
            const std::size_t partials = leased_partials(recorder, scratch_elements);
            const ScratchBuffer scratch(recorder.scratch_pool(),
                                        partials * sizeof(int64_t),
                                        reduction_device_index(t, result));
            operands.scratch(scratch);

            TaskDescription description;
            description.name = name;
            description.operation = op_id;
            description.shape = ReductionShape{size, 1, t.dtype};

            TaskHandle handle;

            // REAL only: an integer or complex tensor is rejected by name; see modules/operations/MATH_README.md.
            const Detail::CountingReductionArm<std::remove_reference_t<Dispatcher>> arm{
                recorder, description, operands, dispatch, size, read_T, write_R, scratch, partials,
                handle};
            with_element_type<Domain::Real>(t.dtype, "Reduction", name, arm);

            return handle;
        }

        /**
         * @brief Folds a whole tensor as `finalize(fold(transform(t[i])), n)`, on the device or the host.
         * @tparam Accum The Reduce accumulator the fold runs in.
         * @tparam Finalize Returns its value in the type it is handed: double on the host path of a sum.
         * @see modules/operations/MATH_README.md
         */
        template<template<typename> class Accum, typename Transform, typename Finalize>
        class FoldingReductionDispatch final
        {
            public:
                /** @brief Holds the callable applied to each element and the one applied to the fold. */
                FoldingReductionDispatch(Transform transform, Finalize finalize)
                    : transform_(std::move(transform)), finalize_(std::move(finalize))
                {

                }

                /** @brief Submits the fixed-order SYCL fold and returns the events of its kernels. */
                template<typename T, typename W>
                [[nodiscard]] Device::SubmittedEvents operator()(
                    sycl::queue& q, const T* in, T* out, int64_t size, ReductionScratch<W> scratch,
                    const std::vector<sycl::event>& deps) const
                {
                    return Reduce::reduce<Accum>(q, in, out, static_cast<std::size_t>(size),
                                                 scratch.data, scratch.capacity,
                                                 transform_, finalize_, deps);
                }

                /** @brief Runs the serial host fold in index order. */
                template<typename T>
                void reference(const T* in, T* out, int64_t size) const
                {
                    Detail::require_finalize_keeps_width<
                        Finalize, typename Serial::Fold<Accum, T>::Wide>();

                    Serial::reduce<Accum>(in, out, static_cast<std::size_t>(size), transform_,
                                          finalize_);
                }

            private:
                /// @brief The callable applied to each element before the fold.
                Transform transform_;

                /// @brief The callable applied once to the folded value.
                Finalize finalize_;
        };

        /**
         * @brief Records a whole-tensor reduction as `result = finalize(fold(transform(t[i])), n)`.
         * @tparam Accum The Reduce accumulator the fold runs in.
         * @param transform Callable `(auto x) -> auto`, applied to each element before the fold.
         * @param finalize Callable `(auto value, std::size_t n) -> auto`, applied once; value is double on the host path of a sum.
         * @throws std::invalid_argument as execute_reduction does.
         * @see modules/operations/MATH_README.md
         */
        template<template<typename> class Accum, typename Transform, typename Finalize>
        TaskHandle execute_folding_reduction(TaskRecorder& recorder, const Tensor& t,
                                             Tensor& result, const char* name,
                                             SushiRuntime::OperationID op_id, Transform transform,
                                             Finalize finalize)
        {
            require_storage(t, name, "t");
            require_device_dtype(t, name, "t");
            require_storage(result, name, "result");
            require_device_dtype(result, name, "result");

            return execute_reduction(
                recorder, t, result, name, op_id,
                FoldingReductionDispatch<Accum, Transform, Finalize>(std::move(transform),
                                                                     std::move(finalize)),
                planned_partials_for(t));
        }

        /**
         * @brief Resolves a rank-2 tensor and an axis into the stride pair a segmented reduction walks.
         * @param axis 0 reduces down the columns, 1 along the rows.
         * @param result Rank-1 tensor with one element per surviving position.
         * @throws std::invalid_argument on a rank, axis or length mismatch.
         * @see modules/operations/MATH_README.md
         */
        [[nodiscard]] inline Reduce::AxisSpan axis_span_for(const Tensor& t, int axis,
                                                            const Tensor& result,
                                                            const char* name)
        {
            require_rank(t, 2, name, "t");
            require_operand(axis == 0 || axis == 1, name, "axis",
                            "is {}; a rank-2 operand has axes 0 and 1", axis);
            require_rank(result, 1, name, "result");

            // The surviving axis is the one not folded away; its extent is the number of folds; see modules/operations/MATH_README.md.
            const int kept = 1 - axis;

            require_extent(result.num_elements, t.shape[kept], "elements", name, "result");

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
         * @throws std::invalid_argument on a rank or shape mismatch, or a row-sliced view.
         * @see modules/operations/MATH_README.md
         */
        [[nodiscard]] inline Reduce::AxisSpan spatial_span_for(const Tensor& t, const Tensor& result,
                                                               const char* name)
        {
            require_rank(t, 4, name, "t");
            require_rank(result, 2, name, "result");
            require_extent(result.shape[0], t.shape[0], "rows", name, "result");
            require_extent(result.shape[1], t.shape[3], "columns", name, "result");
            require_operand(t.strides[1] == t.strides[2] * t.shape[2], name, "t",
                            "is a row-sliced view: row stride {}, column stride {}, width {}; "
                            "materialise it first",
                            t.strides[1], t.strides[2], t.shape[2]);

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
         * @param finalize Callable `(auto value, std::size_t n) -> auto`, applied once per segment; value is double on the host path of a sum.
         * @throws std::invalid_argument on an operand violation; UnsupportedExecution on a device-only tensor under ForceReference or a dtype a tensor's device cannot compute in.
         * @see modules/operations/MATH_README.md
         */
        template<template<typename> class Accum, typename Transform, typename Finalize>
        TaskHandle execute_folding_span_reduction(TaskRecorder& recorder, const Tensor& t,
                                                  const Reduce::AxisSpan& span, Tensor& result,
                                                  const char* name,
                                                  SushiRuntime::OperationID op_id,
                                                  Transform transform, Finalize finalize)
        {
            require_storage(t, name, "t");
            require_device_dtype(t, name, "t");
            require_storage(result, name, "result");
            require_device_dtype(result, name, "result");
            require_dtype(result, t.dtype, name, "result");
            require_domain<Domain::Real>(t.dtype, "Axis reduction", name);

            if (result.num_elements == 0)
                return TaskHandle();

            const void* read_T = t.data_or_null();
            void* write_R = result.data();

            // Operands are declared by Tensor::region(), not by data(); see modules/operations/MATH_README.md.
            OperandSet operands;
            operands.reads(t).writes(result);

            TaskDescription description;
            description.name = name;
            description.operation = op_id;

            TaskHandle handle;

            // REAL only: a fold needs an ordering and an identity, and integers are not arithmetic here; see modules/operations/MATH_README.md.
            const Detail::SpanReductionArm<Accum, Transform, Finalize> arm{
                recorder, description, operands, span, read_T, write_R, transform, finalize,
                handle};
            with_element_type<Domain::Real>(t.dtype, "Axis reduction", name, arm);

            return handle;
        }

        /**
         * @brief Records a segmented reduction over one axis of a rank-2 tensor.
         * @param axis Which axis to fold away; see axis_span_for.
         * @param result Rank-1 tensor sharing t's dtype.
         * @throws std::invalid_argument for the reasons axis_span_for and execute_folding_span_reduction list.
         * @see modules/operations/MATH_README.md
         */
        template<template<typename> class Accum, typename Transform, typename Finalize>
        TaskHandle execute_folding_axis_reduction(TaskRecorder& recorder, const Tensor& t, int axis,
                                                  Tensor& result, const char* name,
                                                  SushiRuntime::OperationID op_id,
                                                  Transform transform, Finalize finalize)
        {
            require_storage(t, name, "t");
            require_device_dtype(t, name, "t");
            require_storage(result, name, "result");
            require_device_dtype(result, name, "result");
            require_dtype(result, t.dtype, name, "result");

            const Reduce::AxisSpan span = axis_span_for(t, axis, result, name);

            return execute_folding_span_reduction<Accum>(recorder, t, span, result, name, op_id,
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
