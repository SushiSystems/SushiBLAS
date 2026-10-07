/**************************************************************************/
/* task_metadata.hpp                                                      */
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
 * @file task_metadata.hpp
 * @brief Declares BlasMetadata, which fills a BLAS task's metadata slots in one fixed order.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/engine/blas/scalar.hpp>
#include <SushiRuntime/graph/task_types.hpp>

namespace SushiBLAS
{
    namespace Internal
    {
        /**
         * @brief Builds the TaskMetadata of one BLAS task: scalars, then flags, then extents.
         * @see include/SushiBLAS/engine/blas/README.md
         */
        class BlasMetadata final
        {
            public:
                /**
                 * @brief Names the task and its operation.
                 * @param name Must outlive the recorded task; a string literal does.
                 */
                BlasMetadata(const char* name, SushiRuntime::Graph::OpID op_id) noexcept
                {
                    meta_.name = name;
                    meta_.task_type = SushiRuntime::Graph::TaskType::MATH_OP;
                    meta_.op_id = op_id;
                }

                /** @brief Appends a scalar as two slots: its real part, then its imaginary part. */
                BlasMetadata& scalar(const BlasScalar& value) noexcept
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
                BlasMetadata& flag(Flag value) noexcept
                {
                    push(static_cast<uint64_t>(value));
                    return *this;
                }

                /** @brief Appends an extent of the call, such as a band width, as one slot. */
                BlasMetadata& extent(int64_t value) noexcept
                {
                    push(value);
                    return *this;
                }

                /** @brief Returns the metadata built so far. */
                [[nodiscard]] const SushiRuntime::Graph::TaskMetadata& get() const noexcept
                {
                    return meta_;
                }

            private:
                /** @brief Writes one value to the next free slot. */
                template<typename Value>
                void push(Value value) noexcept
                {
                    SB_ASSERT(next_ < static_cast<int>(
                                          SushiRuntime::Graph::TaskMetadata::K_MAX_PARAMS),
                              "A BLAS task declared more metadata slots than TaskMetadata holds");
                    meta_.set_param(next_, value);
                    ++next_;
                }

                /// @brief The metadata being built.
                SushiRuntime::Graph::TaskMetadata meta_;

                /// @brief The next free slot.
                int next_ = 0;
        };
    } // namespace Internal
} // namespace SushiBLAS
