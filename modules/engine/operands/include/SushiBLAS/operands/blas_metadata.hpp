/**************************************************************************/
/* blas_metadata.hpp                                                      */
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
 * @file blas_metadata.hpp
 * @brief Declares BLASMetadata, which fills a BLAS task's metadata slots in one fixed order.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/operands/operands_log_category.hpp>
#include <SushiBLAS/operands/scalar.hpp>
#include <SushiBLAS/cost/cost_shape.hpp>
#include <SushiBLAS/engine/task_description.hpp>
#include <SushiRuntime/core/operation_id.hpp>

namespace SushiBLAS
{
    namespace Internal
    {
        /**
         * @brief Builds the TaskDescription of one BLAS task: scalars, then flags, then extents.
         * @see modules/operations/blas/README.md
         */
        class BLASMetadata final
        {
            public:
                /**
                 * @brief Names the task and its operation.
                 * @param name Must outlive the recorded task; a string literal does.
                 */
                BLASMetadata(const char* name, SushiRuntime::OperationID op_id) noexcept
                {
                    description_.name = name;
                    description_.operation = op_id;
                }

                /** @brief Appends a scalar as two slots: its real part, then its imaginary part. */
                BLASMetadata& scalar(const BLASScalar& value) noexcept
                {
                    push(value.real());
                    push(value.imag());
                    return *this;
                }

                /**
                 * @brief Appends a flag as one slot holding its numeric value.
                 * @tparam Flag Core::Transpose, Core::Uplo, Core::Diag or Core::Side.
                 */
                template<typename Flag>
                BLASMetadata& flag(Flag value) noexcept
                {
                    push(static_cast<uint64_t>(value));
                    return *this;
                }

                /** @brief Appends an extent of the call, such as a band width, as one slot. */
                BLASMetadata& extent(int64_t value) noexcept
                {
                    push(value);
                    return *this;
                }

                /** @brief Sets the shape facts the cost rule of the operation reads. */
                BLASMetadata& shape(const CostShape& facts) noexcept
                {
                    description_.shape = facts;
                    return *this;
                }

                /** @brief Returns the metadata built so far. */
                [[nodiscard]] const TaskDescription& get() const noexcept
                {
                    return description_;
                }

            private:
                /** @brief Writes one value to the next free slot. */
                template<typename Value>
                void push(Value value) noexcept
                {
                    SB_ASSERT(K_OPERANDS_LOG_CATEGORY, next_ < static_cast<int>(TaskDescription::K_MAX_PARAMETERS),
                              "A BLAS task declared more slots than TaskDescription holds");
                    description_.set_parameter(next_, value);
                    ++next_;
                }

                /// @brief The metadata being built.
                TaskDescription description_;

                /// @brief The next free slot.
                int next_ = 0;
        };
    } // namespace Internal
} // namespace SushiBLAS
