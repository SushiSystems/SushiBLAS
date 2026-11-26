/**************************************************************************/
/* gemm_tile_shape.hpp                                                    */
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
 * @file gemm_tile_shape.hpp
 * @brief Declares GemmTileShape, the compile-time geometry of one tiled GEMM kernel.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <sycl/sycl.hpp>
#include <SushiBLAS/arithmetic/accumulator.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        /**
         * @brief Returns the largest power-of-two square register block of T that fits a byte budget.
         * @param budget_bytes Bytes of private memory the block may occupy.
         * @param cap Upper bound on the block edge, whatever the budget allows.
         * @return An edge length in [1, cap].
         */
        template<typename T>
        [[nodiscard]] inline constexpr int register_block_dim(std::size_t budget_bytes,
                                                              int cap) noexcept
        {
            int dim = 1;
            while (dim * 2 <= cap &&
                   static_cast<std::size_t>(dim * 2) *
                   static_cast<std::size_t>(dim * 2) * sizeof(T) <= budget_bytes)
                dim *= 2;
            return dim;
        }

        /**
         * @brief Holds the largest register block edge a tiled GEMM work-item accumulates.
         * @see modules/kernels/blas_seam_kernels/README.md
         */
        inline constexpr int K_GEMM_TILE_ITEM_CAP = 8;

        /** @brief Holds the elements of T one 128-bit load reads; 1 marks a scalar-read type. */
        template<typename T>
        struct GemmVectorLanes
        {
            /** @brief Elements one load reads. */
            static constexpr int K_VALUE = 1;
        };

        /** @brief Holds the four float lanes of one 128-bit load. */
        template<>
        struct GemmVectorLanes<float>
        {
            /** @brief Elements one load reads. */
            static constexpr int K_VALUE = 4;
        };

        /** @brief Holds the two double lanes of one 128-bit load. */
        template<>
        struct GemmVectorLanes<double>
        {
            /** @brief Elements one load reads. */
            static constexpr int K_VALUE = 2;
        };

        /** @brief Holds the eight HALF lanes of one 128-bit load. */
        template<>
        struct GemmVectorLanes<sycl::half>
        {
            /** @brief Elements one load reads. */
            static constexpr int K_VALUE = 8;
        };

        /**
         * @brief Returns the elements of a register block edge one local load reads together.
         * @param item_dim The register block edge, a power of two.
         * @return The vector lanes of T, capped at item_dim.
         */
        template<typename T>
        [[nodiscard]] inline constexpr int gemm_fragment_dim(int item_dim) noexcept
        {
            return GemmVectorLanes<T>::K_VALUE < item_dim ? GemmVectorLanes<T>::K_VALUE : item_dim;
        }

        /**
         * @brief Holds the geometry of one tiled GEMM kernel for element type T.
         * @tparam WgRows Work-items down the row axis of a work-group.
         * @tparam WgCols Work-items across the column axis of a work-group.
         * @tparam AccumulatorBytes Private bytes one work-item's register block may occupy.
         * @tparam TileK K-extent staged into local memory per iteration.
         * @see modules/kernels/blas_seam_kernels/README.md
         */
        template<typename T, int WgRows, int WgCols, std::size_t AccumulatorBytes, int TileK>
        struct GemmTileShape
        {
            /** @brief Work-items down the row axis of a work-group. */
            static constexpr int K_WG_ROWS = WgRows;

            /** @brief Work-items across the column axis of a work-group. */
            static constexpr int K_WG_COLS = WgCols;

            /** @brief Work-items per work-group. */
            static constexpr int K_WG_ITEMS = K_WG_ROWS * K_WG_COLS;

            /** @brief Rows of C accumulated privately by one work-item. */
            static constexpr int K_ITEM_ROWS =
                register_block_dim<Accumulator<T>>(AccumulatorBytes, K_GEMM_TILE_ITEM_CAP);

            /** @brief Columns of C accumulated privately by one work-item. */
            static constexpr int K_ITEM_COLS = K_ITEM_ROWS;

            /** @brief Rows of C covered by one work-group. */
            static constexpr int K_BLOCK_M = K_WG_ROWS * K_ITEM_ROWS;

            /** @brief Columns of C covered by one work-group. */
            static constexpr int K_BLOCK_N = K_WG_COLS * K_ITEM_COLS;

            /** @brief K-extent staged into local memory per iteration. */
            static constexpr int K_TILE_K = TileK;

            /**
             * @brief Neighbouring rows or columns of a register block that one local load reads.
             * @see modules/kernels/blas_seam_kernels/README.md
             */
            static constexpr int K_FRAGMENT = gemm_fragment_dim<T>(K_ITEM_ROWS);

            /**
             * @brief Leading dimension of the A tile, one row per K-step, padded by a fragment.
             * @see modules/kernels/blas_seam_kernels/README.md
             */
            static constexpr int K_TILE_A_LD = K_BLOCK_M + K_FRAGMENT;

            /**
             * @brief Leading dimension of the B tile, padded off a bank multiple by a fragment.
             * @see modules/kernels/blas_seam_kernels/README.md
             */
            static constexpr int K_TILE_B_LD = K_BLOCK_N + (K_FRAGMENT > 2 ? K_FRAGMENT : 2);

            static_assert(K_ITEM_ROWS % K_FRAGMENT == 0 && K_ITEM_COLS % K_FRAGMENT == 0,
                          "a register block splits into whole fragments");

            /** @brief Returns the local memory one work-group needs, in bytes. */
            [[nodiscard]] static constexpr std::size_t local_mem_bytes() noexcept
            {
                return (static_cast<std::size_t>(K_TILE_K) * K_TILE_A_LD +
                        static_cast<std::size_t>(K_TILE_K) * K_TILE_B_LD) * sizeof(T);
            }
        };

    } // namespace Kernels
} // namespace SushiBLAS
