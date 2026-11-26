/**************************************************************************/
/* gemm_depth_slice_plan.hpp                                              */
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
 * @file gemm_depth_slice_plan.hpp
 * @brief Declares the plan that cuts the K extent of an under-filled tiled GEMM into slices.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <SushiBLAS/arithmetic/accumulator.hpp>
#include <SushiBLAS/launch/device_policy.hpp>
#include <SushiBLAS/blas_seam_kernels/extent_index.hpp>
#include <SushiBLAS/blas_seam_kernels/level3/gemm_tile_catalog.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        /** @brief Holds the most work-groups per compute unit a depth-sliced launch aims for. */
        inline constexpr int64_t K_MAX_SLICE_GROUPS_PER_UNIT = 4;

        /** @brief Holds the fewest K-steps one depth slice sums. */
        inline constexpr int64_t K_MIN_SLICE_DEPTH = 256;

        /**
         * @brief Holds how a tiled GEMM cuts K into slices and the scratch their partial sums take.
         * @see modules/kernels/blas_seam_kernels/README.md
         */
        struct GemmDepthSlicePlan
        {
            int64_t slices = 0;            /**< Slices per batch entry; 0 when not sliced. */
            int64_t depth = 0;             /**< K-steps per slice, the last one cut at k. */
            std::size_t scratch_bytes = 0; /**< Bytes of the partial sums of every slice. */
        };

        /** @brief Holds the extents and the device facts a depth-slice plan is drawn from. */
        struct GemmDepthSliceProblem
        {
            int64_t m = 0;                 /**< Rows of C. */
            int64_t n = 0;                 /**< Columns of C. */
            int64_t k = 0;                 /**< The summed extent. */
            int64_t batch = 1;             /**< Matrices of C. */
            int64_t block_m = 1;           /**< Rows of C one work-group owns. */
            int64_t block_n = 1;           /**< Columns of C one work-group owns. */
            int64_t tile_k = 1;            /**< K-steps one work-group stages at a time. */
            int64_t compute_units = 1;     /**< Compute units of the device. */
            int64_t groups_per_unit = 1;   /**< Work-groups one compute unit holds at once. */
            std::size_t element_bytes = 1; /**< Bytes of one element of C. */
        };

        /**
         * @brief Returns the slices that fill the device when the tiles of C alone leave it idle.
         * @return A plan with zero slices when the tiles fill the device or K is too short for two.
         * @see modules/kernels/blas_seam_kernels/README.md
         */
        [[nodiscard]] constexpr GemmDepthSlicePlan plan_depth_slices(
            const GemmDepthSliceProblem& p) noexcept
        {
            const int64_t tiles = Indexing::ceil_div(p.m, p.block_m) *
                                  Indexing::ceil_div(p.n, p.block_n) * p.batch;
            const int64_t per_unit = p.groups_per_unit < K_MAX_SLICE_GROUPS_PER_UNIT
                ? (p.groups_per_unit > 1 ? p.groups_per_unit : int64_t{1})
                : K_MAX_SLICE_GROUPS_PER_UNIT;
            const int64_t wanted = p.compute_units * per_unit;
            if (tiles <= 0 || tiles >= wanted)
                return GemmDepthSlicePlan{};

            const int64_t by_fill = Indexing::ceil_div(wanted, tiles);
            const int64_t by_depth = p.k / K_MIN_SLICE_DEPTH;
            const int64_t count = by_fill < by_depth ? by_fill : by_depth;
            if (count < 2)
                return GemmDepthSlicePlan{};

            const int64_t depth =
                Indexing::ceil_div(Indexing::ceil_div(p.k, count), p.tile_k) * p.tile_k;
            const int64_t slices = Indexing::ceil_div(p.k, depth);
            if (slices < 2)
                return GemmDepthSlicePlan{};

            return GemmDepthSlicePlan{
                slices, depth,
                static_cast<std::size_t>(slices * p.batch * p.m * p.n) * p.element_bytes};
        }

        /**
         * @brief Returns the Balanced work-groups of T one compute unit's local memory holds.
         * @return At least 1.
         */
        template<typename T>
        [[nodiscard]] inline int64_t balanced_groups_per_unit(const Device::Profile& profile) noexcept
        {
            using Balanced = GemmTile<T, GemmTileRole::Balanced>;
            const std::size_t held = profile.local_mem_per_compute_unit / Balanced::local_mem_bytes();
            return held > 1 ? static_cast<int64_t>(held) : int64_t{1};
        }

        /**
         * @brief Returns the depth-slice plan of a GEMM of T on the Balanced tile of a device.
         * @return A plan with zero slices when the device has no local tile for Balanced, T
         *         accumulates wider than it stores, or plan_depth_slices declines.
         * @see modules/kernels/blas_seam_kernels/README.md
         */
        template<typename T>
        [[nodiscard]] GemmDepthSlicePlan plan_gemm_depth_slices(const Device::Profile& profile,
                                                                int64_t m, int64_t n, int64_t k,
                                                                int64_t batch) noexcept
        {
            using Balanced = GemmTile<T, GemmTileRole::Balanced>;

            if (!std::is_same<Accumulator<T>, T>::value || !profile.has_dedicated_local_memory ||
                !Device::hosts_local_tile(profile,
                                          {Balanced::local_mem_bytes(),
                                           static_cast<std::size_t>(Balanced::K_WG_ITEMS)}))
                return GemmDepthSlicePlan{};

            return plan_depth_slices(GemmDepthSliceProblem{
                m, n, k, batch, Balanced::K_BLOCK_M, Balanced::K_BLOCK_N, Balanced::K_TILE_K,
                static_cast<int64_t>(profile.compute_units),
                balanced_groups_per_unit<T>(profile), sizeof(T)});
        }

        /** @brief Returns whether scratch of scratch_bytes holds the partial sums plan asks for. */
        [[nodiscard]] inline bool depth_slices_are_ready(const GemmDepthSlicePlan& plan,
                                                         const void* scratch,
                                                         std::size_t scratch_bytes) noexcept
        {
            return plan.slices >= 2 && scratch != nullptr && scratch_bytes >= plan.scratch_bytes;
        }
    } // namespace Kernels
} // namespace SushiBLAS
