/**************************************************************************/
/* task.hpp                                                               */
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
 * @file task.hpp
 * @brief Declares the recorder of one dtype-dispatched task that runs a kernel pair.
 * @author Mustafa Garip
 */

#pragma once

#include <vector>
#include <complex>
#include <cstdint>
#include <utility>
#include <type_traits>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/core/traits.hpp>
#include <SushiBLAS/engine/dispatch.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/task_handle.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiBLAS/engine/kernel_pair_dispatcher.hpp>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiBLAS
{
    namespace Internal
    {
        namespace Detail
        {
            /**
             * @brief Calls a task body with a value of the element type its dtype names.
             * @tparam Func Callable `(T, const KernelPairDispatcher&, sycl::queue&,
             *         const std::vector<sycl::event>&) -> sycl::event` for each T in Dom.
             * @see include/SushiBLAS/engine/README.md
             */
            template<Dispatch::Domain Dom, typename Func>
            class TypedTaskBody
            {
                public:
                    /**
                     * @brief Holds everything the body needs at execute time.
                     * @param family Stored, not copied; must outlive the task.
                     * @param name Stored, not copied; must outlive the task, as TaskMetadata::name must.
                     */
                    TypedTaskBody(Core::DataType dtype, const char* family, const char* name,
                                  Func op_func)
                        : dtype_(dtype), family_(family), name_(name), op_func_(std::move(op_func))
                    {

                    }

                    /**
                     * @brief Runs the body for the recorded dtype.
                     * @throws std::runtime_error if dtype falls outside Dom.
                     */
                    sycl::event operator()(const KernelPairDispatcher& dispatcher, sycl::queue& q,
                                           const std::vector<sycl::event>& deps) const
                    {
                        // The seven arms mirror Dispatch::with_element_type; see include/SushiBLAS/engine/README.md.
                        switch (dtype_)
                        {
                            case Core::DataType::Half:
                                return op_func_(sycl::half{}, dispatcher, q, deps);
                            case Core::DataType::Float32:
                                return op_func_(float{}, dispatcher, q, deps);
                            case Core::DataType::Float64:
                                return op_func_(double{}, dispatcher, q, deps);
                            case Core::DataType::Complex32:
                                if constexpr (Dispatch::domain_admits(Dom, Dispatch::Domain::Complex))
                                    return op_func_(std::complex<float>{}, dispatcher, q, deps);
                                break;
                            case Core::DataType::Complex64:
                                if constexpr (Dispatch::domain_admits(Dom, Dispatch::Domain::Complex))
                                    return op_func_(std::complex<double>{}, dispatcher, q, deps);
                                break;
                            case Core::DataType::Int32:
                                if constexpr (Dispatch::domain_admits(Dom, Dispatch::Domain::Integral))
                                    return op_func_(int32_t{}, dispatcher, q, deps);
                                break;
                            case Core::DataType::Int64:
                                if constexpr (Dispatch::domain_admits(Dom, Dispatch::Domain::Integral))
                                    return op_func_(int64_t{}, dispatcher, q, deps);
                                break;
                        }

                        // The switch ends in a throw, so a dtype that slips past require_domain cannot no-op; see include/SushiBLAS/engine/README.md.
                        SB_THROW_IF(true, "{} '{}' does not support dtype {}.",
                                    family_, name_, Core::to_string(dtype_));
                        return sycl::event();
                    }

                private:
                    /// @brief The dtype the task was recorded for.
                    Core::DataType dtype_;

                    /// @brief The operation family, for diagnostics.
                    const char* family_;

                    /// @brief The operation name, for diagnostics.
                    const char* name_;

                    /// @brief The body the operation recorded.
                    Func op_func_;
            };
        } // namespace Detail

        /**
         * @brief Records one dtype-dispatched task whose body runs a reference and SYCL kernel pair.
         * @param family What kind of operation this is, for diagnostics; a string literal.
         * @param op_func Callable taking a scalar of the element type, a const
         *        KernelPairDispatcher&, the queue and the dependencies.
         * @throws std::invalid_argument at record time on a dtype outside Dom; UnsupportedExecution as TaskRecorder::record_pair states.
         * @see include/SushiBLAS/engine/README.md
         */
        template<Dispatch::Domain Dom = Dispatch::Domain::RealAndComplex, typename Func>
        TaskHandle execute_typed_task(TaskRecorder& recorder,
                                      const SushiRuntime::Graph::TaskMetadata& meta,
                                      const OperandSet& operands, Core::DataType dtype,
                                      const char* family, Func&& op_func)
        {
            Dispatch::require_domain<Dom>(dtype, family, meta.name);

            return recorder.record_pair(meta, operands, family,
                                        Detail::TypedTaskBody<Dom, std::decay_t<Func>>(
                                            dtype, family, meta.name, std::forward<Func>(op_func)));
        }
    } // namespace Internal
} // namespace SushiBLAS
