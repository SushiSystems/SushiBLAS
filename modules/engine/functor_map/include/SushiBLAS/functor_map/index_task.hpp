/**************************************************************************/
/* index_task.hpp                                                         */
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

/**
 * @file index_task.hpp
 * @brief Declares the task bodies of the per-index seam, with their serial host path and their SYCL path.
 * @author Mustafa Garip
 */

#pragma once

#include <string>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <type_traits>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/functor_map/functor_map_log_category.hpp>
#include <SushiBLAS/engine/dispatch.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/task_description.hpp>
#include <SushiBLAS/engine/task_handle.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiBLAS/launch/launch_1d.hpp>
#include <SushiBLAS/reference/serial_1d.hpp>
#include <SushiBLAS/engine/kernel_pair_dispatcher.hpp>
#include <SushiBLAS/functor_map/read_binding.hpp>

namespace SushiBLAS
{
    namespace Dispatch
    {
        namespace Detail
        {
            /**
             * @brief Runs an index loop on the host for the element type of a with_element_type arm.
             * @tparam Op What Detail::task_functor returned for the seam's functor.
             */
            template<typename Op>
            struct SerialIndexArm
            {
                /** @brief Holds the functor the task carries. */
                const Op& op;
                /** @brief Holds the number of indices. */
                std::size_t n;

                /** @brief Runs the loop for the element type named by the tag. */
                template<typename Tag>
                void operator()(Tag tag) const
                {
                    Serial::indexed(n, kernel_op(tag, op));
                }
            };

            /**
             * @brief Launches an index kernel on the device for the element type of a with_element_type arm.
             * @tparam Op What Detail::task_functor returned for the seam's functor.
             */
            template<typename Op>
            struct LaunchIndexArm
            {
                /** @brief Holds the command group the kernel is launched in. */
                sycl::handler& h;
                /** @brief Holds the queue whose device plans the launch shape. */
                const sycl::queue& q;
                /** @brief Holds the functor the task carries. */
                const Op& op;
                /** @brief Holds the number of indices. */
                std::size_t n;

                /** @brief Launches the kernel for the element type named by the tag. */
                template<typename Tag>
                void operator()(Tag tag) const
                {
                    Launch::indexed(h, q, n, kernel_op(tag, op));
                }
            };

            /**
             * @brief Runs `op(i)` for each index below a fixed count on the host or the device, as its mode says.
             * @tparam Dom The element families the functor is well-formed for.
             * @tparam Op What Detail::task_functor returned for the seam's functor.
             * @see docs/design/LAYERED_EXECUTION/README.md
             */
            template<Domain Dom, typename Op>
            class IndexTask final
            {
                public:
                    /**
                     * @brief Holds everything the body needs at execute time.
                     * @param family Stored, not copied; must outlive the task.
                     */
                    IndexTask(Core::DataType dtype, const char* family, std::string name,
                              std::size_t n, Op op)
                        : dtype_(dtype), family_(family), name_(std::move(name)), n_(n),
                          op_(std::move(op))
                    {

                    }

                    /** @brief Runs the indices in order under ForceReference, else submits the kernel. */
                    sycl::event operator()(const KernelPairDispatcher& pair,
                                           sycl::queue& q,
                                           const std::vector<sycl::event>& deps) const
                    {
                        SB_LOG_TRACE(K_FUNCTOR_MAP_LOG_CATEGORY, "{} '{}': {} indices", family_,
                                     name_, n_);

                        return pair.run(q, deps,
                            [this]
                            {
                                with_element_type<Dom>(dtype_, family_, name_.c_str(),
                                    SerialIndexArm<Op>{op_, n_});
                            },
                            [this](sycl::queue& q2,
                                   const std::vector<sycl::event>& deps2) -> sycl::event
                            {
                                return q2.submit([&](sycl::handler& h)
                                {
                                    h.depends_on(deps2);
                                    with_element_type<Dom>(dtype_, family_, name_.c_str(),
                                        LaunchIndexArm<Op>{h, q2, op_, n_});
                                });
                            });
                    }

                private:
                    /// @brief The dtype the element bindings of the task were recorded for.
                    Core::DataType dtype_;

                    /// @brief The operation family, for diagnostics.
                    const char* family_;

                    /// @brief The operation name, for diagnostics.
                    std::string name_;

                    /// @brief The number of indices.
                    std::size_t n_;

                    /// @brief The functor, with the addresses of its declared reads and writes.
                    Op op_;
            };

            /**
             * @brief Runs a functor for each index below the width a node wrote, on the host.
             * @tparam Fn Callable `(std::size_t i)`, copyable.
             */
            template<typename Fn>
            class LateIndexHostRun final
            {
                public:
                    /**
                     * @brief Holds the functor, the capacity and the address of the width.
                     * @param width The first element of an INT32 tensor in host-readable memory.
                     */
                    LateIndexHostRun(Fn fn, std::string name, std::size_t capacity,
                                     const std::int32_t* width)
                        : fn_(std::move(fn)), name_(std::move(name)), capacity_(capacity),
                          width_(width)
                    {

                    }

                    /**
                     * @brief Reads the width and runs the indices below it in order.
                     * @throws std::runtime_error for a width below 0 or above the capacity, as the
                     *         runtime refuses on the device path.
                     */
                    void operator()() const
                    {
                        const std::int32_t live = *width_;
                        SB_THROW_IF(live < 0 || static_cast<std::size_t>(live) > capacity_,
                                    "Index operation '{}': the late count {} is outside [0, {}]",
                                    name_, live, capacity_);
                        SB_LOG_TRACE(K_FUNCTOR_MAP_LOG_CATEGORY,
                                     "Index operation '{}': {} of {} indices", name_, live,
                                     capacity_);
                        Serial::indexed(static_cast<std::size_t>(live), fn_);
                    }

                private:
                    /// @brief The functor each index calls.
                    Fn fn_;

                    /// @brief The operation name, for diagnostics.
                    std::string name_;

                    /// @brief The most indices the width may name.
                    std::size_t capacity_;

                    /// @brief The width, read when the task runs.
                    const std::int32_t* width_;
            };

            /**
             * @brief Gives a functor the one-index call shape the runtime's dynamic node checks for.
             * @tparam Fn Callable `(std::size_t i)`, trivially copyable.
             */
            template<typename Fn>
            class IndexKernel final
            {
                public:
                    /** @brief Holds the functor. */
                    explicit IndexKernel(Fn fn) : fn_(std::move(fn))
                    {

                    }

                    /** @brief Calls the functor for one index. */
                    void operator()(std::size_t i) const
                    {
                        fn_(i);
                    }

                private:
                    /// @brief The functor each index calls.
                    Fn fn_;
            };

            /**
             * @brief Records a late-count index task for the element type of a with_element_type arm.
             * @tparam Op What Detail::task_functor returned for the seam's functor.
             */
            template<typename Op>
            struct RecordLateIndexArm
            {
                /** @brief Holds the recorder the task goes to. */
                TaskRecorder& recorder;
                /** @brief Holds what the task states about itself. */
                const TaskDescription& description;
                /** @brief Holds the tensors the task reads and writes. */
                const OperandSet& operands;
                /** @brief Holds the operation family, for diagnostics. */
                const char* family;
                /** @brief Holds the tensor that holds the width. */
                const Tensor& count;
                /** @brief Holds the most indices the width may name. */
                std::size_t capacity;
                /** @brief Holds the functor the task carries. */
                const Op& op;
                /** @brief Receives the handle of the recorded task. */
                TaskHandle& handle;

                /** @brief Records the task for the element type named by the tag. */
                template<typename Tag>
                void operator()(Tag tag) const
                {
                    const auto fn = kernel_op(tag, op);
                    handle = recorder.record_late_pair(
                        description, operands, family, count, capacity,
                        LateIndexHostRun<std::decay_t<decltype(fn)>>(
                            fn, description.name, capacity,
                            static_cast<const std::int32_t*>(count.data())),
                        IndexKernel<std::decay_t<decltype(fn)>>(fn));
                }
            };

            /**
             * @brief Records one per-index task over n indices, or over a capacity of n when count is set.
             * @tparam Dom The element families the functor is well-formed for.
             * @tparam Op What Detail::task_functor returned for the seam's functor.
             * @param count The tensor that holds the live width, or null for a fixed count.
             * @return The handle of the recorded task.
             * @throws std::invalid_argument as TaskRecorder::record_late_pair for a bad count tensor.
             */
            template<Domain Dom, typename Op>
            [[nodiscard]] inline TaskHandle record_index_task(
                TaskRecorder& recorder, const TaskDescription& description,
                const OperandSet& operands, Core::DataType dtype, const char* family,
                std::size_t n, const Tensor* count, const Op& op)
            {
                if (count == nullptr)
                    return recorder.record_pair(
                        description, operands, family,
                        IndexTask<Dom, Op>(dtype, family, description.name, n, op));

                TaskHandle handle;
                with_element_type<Dom>(
                    dtype, family, description.name,
                    RecordLateIndexArm<Op>{recorder, description, operands, family, *count, n, op,
                                           handle});
                return handle;
            }
        } // namespace Detail
    } // namespace Dispatch
} // namespace SushiBLAS
