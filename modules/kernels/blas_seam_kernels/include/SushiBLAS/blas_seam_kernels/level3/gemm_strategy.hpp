/**************************************************************************/
/* gemm_strategy.hpp                                                      */
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
 * @file gemm_strategy.hpp
 * @brief Declares the request a caller passes to pick the SYCL GEMM kernel instead of leaving the choice to the device profile.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/blas_seam_kernels/level3/gemm_tile_catalog.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        /**
         * @brief Names the kernel family a GEMM call runs.
         * @see modules/kernels/blas_seam_kernels/README.md
         */
        enum class GemmStrategyKind : std::uint8_t
        {
            Automatic, /**< The device profile selects the kernel; the default. */
            NoLocal,   /**< The kernel that reads its operands from global memory. */
            LocalTiled /**< The kernel that stages tiles through local memory, in the shape of the role. */
        };

        /**
         * @brief Holds the kernel a caller asks for: the family and, for the tiled family, the tile role.
         * @see modules/kernels/blas_seam_kernels/README.md
         */
        struct GemmStrategy
        {
            GemmStrategyKind kind = GemmStrategyKind::Automatic; /**< The kernel family. */
            GemmTileRole role = GemmTileRole::Balanced;          /**< The tile shape of the tiled family. */

            /** @brief Returns the request that leaves the choice to the device profile. */
            [[nodiscard]] static constexpr GemmStrategy automatic() noexcept
            {
                return {};
            }

            /** @brief Returns the request for the kernel that uses no local memory. */
            [[nodiscard]] static constexpr GemmStrategy no_local() noexcept
            {
                return {GemmStrategyKind::NoLocal, GemmTileRole::Balanced};
            }

            /** @brief Returns the request for the tiled kernel in the shape of a role. */
            [[nodiscard]] static constexpr GemmStrategy local_tiled(
                GemmTileRole role = GemmTileRole::Balanced) noexcept
            {
                return {GemmStrategyKind::LocalTiled, role};
            }
        };
    } // namespace Kernels
} // namespace SushiBLAS
