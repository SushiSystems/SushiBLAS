/**************************************************************************/
/* gemm_tile_table.hpp                                                    */
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
 * @file gemm_tile_table.hpp
 * @brief Declares the measured tile choices of the tiled GEMM, keyed by device and product.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/launch/device_policy.hpp>
#include <SushiBLAS/blas_seam_kernels/level3/gemm_tile_catalog.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        /**
         * @brief Names how full one product leaves the device.
         * @see modules/kernels/blas_seam_kernels/README.md
         */
        enum class GemmSizeClass : std::uint8_t
        {
            Small, /**< Fewer Balanced output tiles than compute units. */
            Large  /**< Every compute unit has at least one Balanced output tile. */
        };

        /**
         * @brief Holds one measured choice: the role that won on one architecture for one dtype and size class.
         * @see modules/kernels/blas_seam_kernels/README.md
         */
        struct GemmTileEntry
        {
            /** @brief The architecture the entry was measured on. */
            Device::Architecture architecture = Device::Architecture::Unknown;

            /** @brief The element type of the product. */
            Core::DataType dtype = Core::DataType::Float32;

            /** @brief The size class of the product. */
            GemmSizeClass size_class = GemmSizeClass::Large;

            /** @brief The role that ran fastest. */
            GemmTileRole role = GemmTileRole::Balanced;
        };

        /**
         * @brief Tells whether no two entries of a table name the same architecture, dtype and size class.
         * @tparam Table A container of GemmTileEntry with `size()` and `operator[]`.
         * @param table The entries to compare.
         * @return false when two entries share a key, so the table holds two answers for one product.
         * @see modules/kernels/blas_seam_kernels/README.md
         */
        template<typename Table>
        constexpr bool has_unique_keys(const Table& table)
        {
            for (std::size_t first = 0; first < table.size(); ++first)
            {
                for (std::size_t second = first + 1; second < table.size(); ++second)
                {
                    if (table[first].architecture == table[second].architecture &&
                        table[first].dtype == table[second].dtype &&
                        table[first].size_class == table[second].size_class)
                        return false;
                }
            }
            return true;
        }

        /**
         * @brief Holds every measured entry; `benchmark_gemm_tiles` prints the lines.
         * @see modules/kernels/blas_seam_kernels/README.md
         */
        inline constexpr std::array<GemmTileEntry, 0> K_GEMM_TILE_TABLE = {};

        static_assert(has_unique_keys(K_GEMM_TILE_TABLE), "No two entries of the shipped table share a key.");

    } // namespace Kernels
} // namespace SushiBLAS
