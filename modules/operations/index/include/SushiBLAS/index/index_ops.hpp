/**************************************************************************/
/* index_ops.hpp                                                          */
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
 * @file index_ops.hpp
 * @brief Declares IndexOps, the per-index operation that runs a functor once for each index.
 * @author Mustafa Garip
 */

#pragma once

#include <string>
#include <cstddef>
#include <optional>
#include <type_traits>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/engine/dispatch.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/operation_options.hpp>
#include <SushiBLAS/engine/task_handle.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiBLAS/engine/task_description.hpp>
#include <SushiBLAS/functor_map/index_task.hpp>
#include <SushiBLAS/functor_map/read_binding.hpp>
#include <SushiBLAS/functor_map/write_binding.hpp>
#include <SushiRuntime/core/operation_id.hpp>

namespace SushiBLAS
{
    /**
     * @brief Records tasks that call one functor for each index below a count.
     * @see modules/operations/index/README.md
     */
    class IndexOps final
    {
        public:
            /** @brief Builds a view that records through recorder, which must outlive the view. */
            explicit IndexOps(TaskRecorder& recorder) noexcept : recorder_(recorder) {}

            IndexOps(const IndexOps&) = default;
            IndexOps& operator=(const IndexOps&) = delete;
            IndexOps(IndexOps&&) = default;
            IndexOps& operator=(IndexOps&&) = delete;
            ~IndexOps() = default;

            /**
             * @brief Records one task that calls `functor(i, reads..., writes...)` for each i below count.
             * @tparam Dom The element families a generic functor is well-formed for.
             * @param bound A bind_writes bundle, which may wrap a bind_reads bundle.
             * @pre Two indices write one element only when the caller orders them, one call per colour.
             * @throws std::invalid_argument on an empty name or a declared-tensor violation.
             */
            template<Dispatch::Domain Dom = Dispatch::Domain::Real, typename Bound>
            TaskHandle for_each(std::size_t count, Bound&& bound, OperationOptions options)
            {
                return record<Dom>(count, nullptr, bound, options);
            }

            /**
             * @brief Records one task over capacity indices that runs the first n, n being the
             *        INT32 a previous node wrote into count.
             * @param count A contiguous INT32 tensor in Residency::Host; the bundle must not write it.
             * @pre n lies in [0, capacity] when the task runs, and the tensors outlive the run.
             * @throws std::invalid_argument as the fixed-count form, and for a bad count tensor.
             */
            template<Dispatch::Domain Dom = Dispatch::Domain::Real, typename Bound>
            TaskHandle for_each(std::size_t capacity, const Tensor& count, Bound&& bound,
                                OperationOptions options)
            {
                return record<Dom>(capacity, &count, bound, options);
            }

        private:
            /** @brief Holds the family name every check and diagnostic of this view reports under. */
            static constexpr const char* K_FAMILY = "Index operation";

            /**
             * @brief Checks and records one per-index task over n indices.
             * @param count The late count tensor, or null for a fixed count.
             */
            template<Dispatch::Domain Dom, typename Bound>
            TaskHandle record(std::size_t n, const Tensor* count, const Bound& bound,
                              const OperationOptions& options)
            {
                static_assert(Dispatch::Detail::IsWriteBound<Bound>::value,
                              "for_each takes a bind_writes bundle: a task that declares no write "
                              "is not ordered against anything.");

                Dispatch::require_operation_options(options, "index.for_each");
                const std::string name(options.name);
                const char* const label = name.c_str();

                const Core::DataType dtype = Dispatch::Detail::element_dtype_of(bound)
                                                 .value_or(nominal_dtype<Dom>());
                Dispatch::require_domain<Dom>(dtype, K_FAMILY, label);
                Dispatch::Detail::require_declared_reads(bound, dtype, label);
                Dispatch::Detail::require_declared_writes(bound, dtype, label);

                if (n == 0)
                    return TaskHandle();

                OperandSet operands;
                if (count != nullptr)
                    operands.reads(*count);
                Dispatch::Detail::declare_reads(bound, operands, label);
                Dispatch::Detail::declare_writes(bound, operands, label);
                if (count != nullptr)
                    require_count_not_written(*count, operands, label);

                TaskDescription description;
                description.name = label;
                description.operation = for_each_operation();
                description.determinism = options.determinism;

                const auto op = Dispatch::Detail::task_functor(bound);
                return Dispatch::Detail::record_index_task<Dom>(recorder_, description, operands,
                                                                dtype, K_FAMILY, n, count, op);
            }

            /**
             * @brief Returns the dtype a task runs under when no binding names the element type.
             * @tparam Dom The first family the domain admits, real before complex before integral.
             */
            template<Dispatch::Domain Dom>
            [[nodiscard]] static constexpr Core::DataType nominal_dtype() noexcept
            {
                if constexpr (Dispatch::domain_admits(Dom, Dispatch::Domain::Real))
                    return Core::DataType::Float32;
                else if constexpr (Dispatch::domain_admits(Dom, Dispatch::Domain::Complex))
                    return Core::DataType::Complex32;
                else
                    return Core::DataType::Int32;
            }

            /** @brief Returns the operation id every per-index task is recorded under. */
            [[nodiscard]] static SushiRuntime::OperationID for_each_operation() noexcept;

            /**
             * @brief Checks that no declared write overlaps the count tensor.
             * @throws std::invalid_argument for a task that writes the count it is sized by.
             */
            static void require_count_not_written(const Tensor& count, const OperandSet& operands,
                                                  const char* name);

            /// @brief The recorder every task of this view is recorded on.
            TaskRecorder& recorder_;
    };
} // namespace SushiBLAS
