/**************************************************************************/
/* arg_axis_reduction.hpp                                                 */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                SushiBLAS                               */
/*                https://github.com/SushiSystems/SushiBLAS               */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file arg_axis_reduction.hpp
 * @brief Declares the executor of an index reduction along one axis through the Reduce arg accumulators.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <SushiBLAS/reduction/reduce_1d.hpp>
#include <SushiBLAS/reduction/reduce_axis.hpp>
#include <SushiBLAS/reductions/dispatch.hpp>

namespace SushiBLAS
{
    namespace Dispatch
    {
        namespace Detail
        {
            /**
             * @brief Reads the elements of one segment as (value, index) candidates.
             * @see modules/operations/README.md
             */
            template<typename T>
            class IndexedAxisCursor final
            {
                public:
                    /** @brief Binds the cursor to a segment's first element. */
                    IndexedAxisCursor(const T* base, std::size_t element_stride)
                        : base_(base), element_stride_(element_stride)
                    {

                    }

                    /** @brief Returns element k of the segment with its index along the axis. */
                    [[nodiscard]] Reduce::IndexedValue<T> operator()(std::size_t k) const noexcept
                    {
                        return Reduce::IndexedValue<T>{base_[k * element_stride_],
                                                       static_cast<std::int64_t>(k)};
                    }

                private:
                    /// @brief The segment's first element.
                    const T* base_;

                    /// @brief Input elements between consecutive elements of the segment.
                    std::size_t element_stride_;
            };

            /**
             * @brief Opens the segments of an AxisSpan as indexed cursors for Reduce::reduce_segments.
             * @see modules/operations/README.md
             */
            template<typename T>
            class IndexedAxisOpen final
            {
                public:
                    /** @brief Binds the opener to an input array and its segment geometry. */
                    IndexedAxisOpen(const T* in, const Reduce::AxisSpan& span)
                        : in_(in), segments_(span.segments), segment_stride_(span.segment_stride),
                          segment_stride2_(span.segment_stride2),
                          element_stride_(span.element_stride)
                    {

                    }

                    /** @brief Returns the cursor of segment s. */
                    [[nodiscard]] IndexedAxisCursor<T> operator()(std::size_t s) const noexcept
                    {
                        const std::size_t s2 = s / segments_;
                        const std::size_t s1 = s % segments_;
                        return IndexedAxisCursor<T>(
                            in_ + s2 * segment_stride2_ + s1 * segment_stride_, element_stride_);
                    }

                private:
                    /// @brief The input's first element.
                    const T* in_;

                    /// @brief Segments per segment group.
                    std::size_t segments_;

                    /// @brief Input elements between the starts of consecutive segments.
                    std::size_t segment_stride_;

                    /// @brief Input elements between the starts of consecutive segment groups.
                    std::size_t segment_stride2_;

                    /// @brief Input elements between consecutive elements of one segment.
                    std::size_t element_stride_;
            };

            /**
             * @brief Stores the winning index of a segment as an INT64 element.
             * @see modules/operations/README.md
             */
            class IndexedAxisStore final
            {
                public:
                    /** @brief Binds the store to an INT64 output array and its stride. */
                    IndexedAxisStore(std::int64_t* out, std::size_t out_stride)
                        : out_(out), out_stride_(out_stride)
                    {

                    }

                    /** @brief Writes the index of the winning candidate of segment s. */
                    template<typename P>
                    void operator()(std::size_t s, P best) const noexcept
                    {
                        out_[s * out_stride_] = best.index;
                    }

                private:
                    /// @brief The output's first element.
                    std::int64_t* out_;

                    /// @brief Output elements between consecutive results.
                    std::size_t out_stride_;
            };

            /**
             * @brief Runs one segmented index fold on the host or the device, as its mode says.
             * @tparam Accum Reduce::RunningArgMax or Reduce::RunningArgMin.
             * @see modules/operations/README.md
             */
            template<template<typename> class Accum, typename T>
            class ArgAxisTask final
            {
                public:
                    /** @brief Holds everything the body needs at execute time. */
                    ArgAxisTask(const Reduce::AxisSpan& span, const T* in, std::int64_t* out)
                        : span_(span), in_(in), out_(out)
                    {

                    }

                    /** @brief Folds each segment serially under ForceReference, else submits the kernel. */
                    sycl::event operator()(const KernelPairDispatcher& pair,
                                           const std::vector<sycl::event>& deps) const
                    {
                        return pair.run(deps,
                            [this]
                            {
                                const IndexedAxisOpen<T> open(in_, span_);
                                const IndexedAxisStore store(out_, span_.out_stride);

                                const std::size_t total = span_.total_segments();
                                for (std::size_t s = 0; s < total; ++s)
                                {
                                    const auto cursor = open(s);
                                    Accum<Reduce::IndexedValue<T>> fold;
                                    for (std::size_t k = 0; k < span_.extent; ++k)
                                        fold.accumulate(cursor(k));

                                    store(s, fold.value());
                                }
                            },
                            [this](sycl::queue& q2,
                                   const std::vector<sycl::event>& deps2) -> sycl::event
                            {
                                const Reduce::AxisPlan geometry =
                                    Reduce::plan_axis(Device::profile_for(q2),
                                                      span_.total_segments(), span_.extent);

                                return Reduce::reduce_segments<Accum, Reduce::IndexedValue<T>>(
                                    q2, span_.total_segments(), span_.extent, geometry,
                                    IndexedAxisOpen<T>(in_, span_),
                                    IndexedAxisStore(out_, span_.out_stride), deps2);
                            });
                    }

                private:
                    /// @brief The segment geometry the fold walks.
                    Reduce::AxisSpan span_;

                    /// @brief The input's base address.
                    const T* in_;

                    /// @brief The result's base address.
                    std::int64_t* out_;
            };

            /**
             * @brief Records an index fold for the element type of a with_element_type arm.
             * @tparam Accum Reduce::RunningArgMax or Reduce::RunningArgMin.
             * @see modules/operations/README.md
             */
            template<template<typename> class Accum>
            struct ArgAxisArm
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
                std::int64_t* write_R;
                /** @brief Receives the handle of the recorded task. */
                TaskHandle& handle;

                /** @brief Records the task for the element type named by the tag. */
                template<typename Tag>
                void operator()(Tag) const
                {
                    using T = typename Tag::type;

                    handle = recorder.record_pair(description, operands, "Axis reduction",
                        ArgAxisTask<Accum, T>(span, static_cast<const T*>(read_T), write_R));
                }
            };
        } // namespace Detail

        /**
         * @brief Records the index an arg accumulator elects along one axis of a rank-2 tensor.
         * @tparam Accum Reduce::RunningArgMax or Reduce::RunningArgMin.
         * @param result Rank-1 INT64 tensor with one element per position axis_span_for keeps.
         * @throws std::invalid_argument on an empty t or an operand violation; UnsupportedExecution on a device-only t under ForceReference.
         * @see modules/operations/README.md
         */
        template<template<typename> class Accum>
        TaskHandle execute_arg_axis_reduction(TaskRecorder& recorder, const Tensor& t, int axis,
                                              Tensor& result, const char* name,
                                              SushiRuntime::OperationID op_id)
        {
            require_storage(t, name, "t");
            require_device_dtype(t, name, "t");
            require_storage(result, name, "result");
            require_device_dtype(result, name, "result");
            require_dtype(result, Core::DataType::Int64, name, "result");
            require_domain<Domain::Real>(t.dtype, "Axis reduction", name);
            require_min_elements(t, 1, name, "t");

            const Reduce::AxisSpan span = axis_span_for(t, axis, result, name);

            const void* read_T = t.data_or_null();
            std::int64_t* write_R = result.data_as<std::int64_t>();

            // Operands are declared by Tensor::region(), not by data(); see modules/operations/MATH_README.md.
            OperandSet operands;
            operands.reads(t).writes(result);

            TaskDescription description;
            description.name = name;
            description.operation = op_id;
            description.shape = ReductionShape{t.num_elements, result.num_elements, t.dtype};

            TaskHandle handle;

            // REAL only: the order the candidates are compared in is the one NaN rule of argmax; see modules/operations/README.md.
            const Detail::ArgAxisArm<Accum> arm{recorder, description, operands, span, read_T,
                                                write_R, handle};
            with_element_type<Domain::Real>(t.dtype, "Axis reduction", name, arm);

            return handle;
        }
    } // namespace Dispatch
} // namespace SushiBLAS
