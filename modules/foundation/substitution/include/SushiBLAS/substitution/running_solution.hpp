/**************************************************************************/
/* running_solution.hpp                                                   */
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
 * @file running_solution.hpp
 * @brief Declares RunningSolution, which carries a substitution's solution unrounded.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <type_traits>
#include <SushiBLAS/arithmetic/accumulator.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        /**
         * @brief Holds the solved elements of a substitution in Accumulator<T> between steps.
         * @see modules/foundation/substitution/README.md
         */
        template<typename T>
        class RunningSolution final
        {
            public:
                /** @brief The type a solved element is carried in. */
                using Wide = Accumulator<T>;

                /** @brief Holds whether Wide differs from T and needs storage of its own. */
                static constexpr bool K_NEEDS_WORKSPACE = !std::is_same_v<Wide, T>;

                /**
                 * @brief Binds the right-hand side and, under K_NEEDS_WORKSPACE, its wide copy.
                 * @param workspace One Wide per element of rounded, counted from workspace_offset;
                 *        never read when K_NEEDS_WORKSPACE is false.
                 */
                RunningSolution(T* rounded, Wide* workspace, int64_t workspace_offset) noexcept
                    : rounded_(rounded), wide_(select(rounded, workspace, workspace_offset))
                {

                }

                /** @brief Returns the solved element at index, unrounded. */
                [[nodiscard]] Wide get(int64_t index) const noexcept
                {
                    return wide_[index];
                }

                /** @brief Stores a solved element, and its rounding in the right-hand side. */
                void set(int64_t index, Wide value) const noexcept
                {
                    wide_[index] = value;
                    if constexpr (K_NEEDS_WORKSPACE)
                        rounded_[index] = static_cast<T>(value);
                }

            private:
                /** @brief Returns where the unrounded solution lives. */
                [[nodiscard]] static Wide* select([[maybe_unused]] T* rounded,
                                                  [[maybe_unused]] Wide* workspace,
                                                  [[maybe_unused]] int64_t offset) noexcept
                {
                    if constexpr (K_NEEDS_WORKSPACE)
                        return workspace + offset;
                    else
                        return rounded;
                }

                /// @brief The right-hand side, which receives every solved element rounded to T.
                T* rounded_;

                /// @brief The unrounded solution: the workspace, or rounded_ when Wide is T.
                Wide* wide_;
        };
    } // namespace Kernels
} // namespace SushiBLAS
