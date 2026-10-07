/**************************************************************************/
/* triangular_plan.hpp                                                    */
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
 * @file triangular_plan.hpp
 * @brief Declares TriangularSolvePlan, the diagonal block size of a triangular solve on a device.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <SushiBLAS/support/device_policy.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        namespace SYCL
        {
            /**
             * @brief Holds the steps one diagonal block takes on a CPU device.
             * @see kernels/README.md
             */
            inline constexpr int64_t K_CPU_SOLVE_BLOCK = 256;

            /**
             * @brief Holds the steps one diagonal block takes on a device that is not a CPU.
             * @see kernels/README.md
             */
            inline constexpr int64_t K_GPU_SOLVE_BLOCK = 128;

            /**
             * @brief Holds how many block widths of otherwise idle work a blocked solve must spread
             *        before it repays its launches.
             * @see kernels/README.md
             */
            inline constexpr int64_t K_SOLVE_BREAK_EVEN_BLOCKS = 2;

            /**
             * @brief Holds how a triangular solve is cut into diagonal blocks.
             * @see kernels/README.md
             */
            struct TriangularSolvePlan
            {
                int64_t block = 1; /**< Steps one diagonal block solves before the rest updates. */

                /** @brief Returns the steps per block for order steps: block within [1, order]. */
                [[nodiscard]] constexpr int64_t block_width(int64_t order) const noexcept
                {
                    const int64_t whole = (order > 0) ? order : 1;
                    if (block < 1)
                        return 1;
                    return (block > whole) ? whole : block;
                }

                /** @brief Returns how many blocks cover order steps; one for an empty system. */
                [[nodiscard]] constexpr int64_t block_count(int64_t order) const noexcept
                {
                    const int64_t count = (order + block_width(order) - 1) / block_width(order);
                    return (count > 0) ? count : 1;
                }
            };

            /**
             * @brief Returns how many work-items a device keeps running at once.
             * @return The compute units of a CPU; the resident work-items of a grid-stride launch
             *         otherwise.
             * @see kernels/README.md
             */
            [[nodiscard]] inline std::size_t resident_work_items(const Device::Profile& p) noexcept
            {
                if (p.is_cpu)
                    return p.compute_units;
                return p.compute_units * Device::K_WAVES_PER_COMPUTE_UNIT *
                       Device::work_group_size(p);
            }

            /**
             * @brief Plans the diagonal blocks of a triangular solve on a device.
             * @param order Steps of one substitution.
             * @param streams Independent substitutions: right-hand-side vectors times batch size.
             * @return One block of the whole order when the substitutions alone fill the device or
             *         the system is too small to repay the launches; the device's block otherwise.
             * @see kernels/README.md
             */
            [[nodiscard]] inline TriangularSolvePlan plan_triangular_solve(const Device::Profile& p,
                                                                          int64_t order,
                                                                          int64_t streams) noexcept
            {
                const TriangularSolvePlan unblocked{(order > 0) ? order : 1};
                const int64_t block = p.is_cpu ? K_CPU_SOLVE_BLOCK : K_GPU_SOLVE_BLOCK;
                const int64_t slots = static_cast<int64_t>(resident_work_items(p));

                if (streams >= slots)
                    return unblocked;
                if (order * (slots - streams) <= K_SOLVE_BREAK_EVEN_BLOCKS * block * slots)
                    return unblocked;
                return TriangularSolvePlan{block};
            }
        } // namespace SYCL
    } // namespace Kernels
} // namespace SushiBLAS
