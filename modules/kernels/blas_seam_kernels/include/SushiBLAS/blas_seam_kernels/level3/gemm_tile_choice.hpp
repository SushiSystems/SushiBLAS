/**************************************************************************/
/* gemm_tile_choice.hpp                                                   */
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
 * @file gemm_tile_choice.hpp
 * @brief Declares choose_gemm_tile, the host rule that picks the tile shape of one tiled GEMM.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/launch/device_policy.hpp>
#include <SushiBLAS/blas_seam_kernels/level3/gemm_tile_catalog.hpp>
#include <SushiBLAS/blas_seam_kernels/level3/gemm_tile_table.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        /**
         * @brief Holds the extents and the element type of one product the chooser sizes.
         * @see modules/kernels/blas_seam_kernels/README.md
         */
        struct GemmTileProduct
        {
            /** @brief Rows of C. */
            std::int64_t m = 0;

            /** @brief Columns of C. */
            std::int64_t n = 0;

            /** @brief Depth of the product. */
            std::int64_t k = 0;

            /** @brief Products in the batch. */
            std::int64_t batch_size = 1;

            /** @brief The element type. */
            Core::DataType dtype = Core::DataType::Float32;
        };

        /**
         * @brief Names what the chooser did with the pin it was given.
         * @see modules/kernels/blas_seam_kernels/README.md
         */
        enum class GemmTilePinStatus : std::uint8_t
        {
            None,    /**< No pin was given. */
            Applied, /**< The pinned role was chosen. */
            Unknown, /**< The pin names no role of the catalog. */
            Refused  /**< The pinned role is past a hard limit of the device. */
        };

        /**
         * @brief Holds the chooser's answer: the role to run, or none when no shape fits the device.
         * @see modules/kernels/blas_seam_kernels/README.md
         */
        struct GemmTileChoice
        {
            /** @brief The role to launch; empty when the device hosts no catalog shape. */
            std::optional<GemmTileRole> role;

            /** @brief What became of the pin. */
            GemmTilePinStatus pin = GemmTilePinStatus::None;
        };

        /**
         * @brief Returns whether a device can launch a shape: local memory, work-group size and whole sub-groups.
         * @see modules/kernels/blas_seam_kernels/README.md
         */
        [[nodiscard]] inline bool admits_gemm_tile(const Device::Profile& profile,
                                                   const GemmTileTraits& shape) noexcept
        {
            if (!profile.has_dedicated_local_memory)
                return false;

            const std::size_t unit_bytes = profile.local_mem_per_compute_unit != 0
                ? profile.local_mem_per_compute_unit
                : profile.local_mem_bytes;
            const std::size_t sub_group = profile.sub_group_size != 0 ? profile.sub_group_size : 1;

            return shape.local_bytes <= profile.local_mem_bytes &&
                   shape.local_bytes <= unit_bytes &&
                   shape.work_group_items <= profile.max_work_group_size &&
                   shape.work_group_items % sub_group == 0;
        }

        /** @brief Returns how many work-groups one shape launches for a product. */
        [[nodiscard]] inline constexpr std::int64_t gemm_output_tiles(
            const GemmTileTraits& shape, const GemmTileProduct& product) noexcept
        {
            const std::int64_t rows = (product.m + shape.block_m - 1) / shape.block_m;
            const std::int64_t cols = (product.n + shape.block_n - 1) / shape.block_n;
            return product.batch_size * rows * cols;
        }

        /** @brief Returns the size class of a product, counted in Balanced output tiles. */
        [[nodiscard]] inline GemmSizeClass gemm_size_class(const Device::Profile& profile,
                                                           const GemmTileTraits& balanced,
                                                           const GemmTileProduct& product) noexcept
        {
            const std::int64_t units = static_cast<std::int64_t>(profile.compute_units);
            return gemm_output_tiles(balanced, product) < units ? GemmSizeClass::Small
                                                                : GemmSizeClass::Large;
        }

        namespace Detail
        {
            /** @brief Returns whether shape a stages fewer bytes per multiply-add than shape b. */
            [[nodiscard]] inline constexpr bool is_denser_tile(const GemmTileTraits& a,
                                                               const GemmTileTraits& b) noexcept
            {
                // Arithmetic per staged element is bm * bn / (bm + bn); compared cross-multiplied.
                return a.block_m * a.block_n * (b.block_m + b.block_n) >
                       b.block_m * b.block_n * (a.block_m + a.block_n);
            }

            /** @brief Returns whether two shapes stage the same bytes per multiply-add. */
            [[nodiscard]] inline constexpr bool is_as_dense_tile(const GemmTileTraits& a,
                                                                 const GemmTileTraits& b) noexcept
            {
                return !is_denser_tile(a, b) && !is_denser_tile(b, a);
            }

            /** @brief Returns whether the model prefers candidate over best for a product. */
            [[nodiscard]] inline bool models_better_tile(const GemmTileTraits& candidate,
                                                         const GemmTileTraits& best,
                                                         const GemmTileProduct& product,
                                                         std::int64_t units) noexcept
            {
                const bool candidate_fills = gemm_output_tiles(candidate, product) >= units;
                const bool best_fills = gemm_output_tiles(best, product) >= units;
                if (candidate_fills != best_fills)
                    return candidate_fills;

                // A product no shape fills takes the shape that spreads it widest.
                if (!candidate_fills)
                {
                    const std::int64_t candidate_tiles = gemm_output_tiles(candidate, product);
                    const std::int64_t best_tiles = gemm_output_tiles(best, product);
                    if (candidate_tiles != best_tiles)
                        return candidate_tiles > best_tiles;
                    return candidate.local_bytes < best.local_bytes;
                }

                if (!is_as_dense_tile(candidate, best))
                    return is_denser_tile(candidate, best);
                return candidate.local_bytes < best.local_bytes;
            }

            /** @brief Returns the sizes of one role from the catalog's traits. */
            [[nodiscard]] inline constexpr const GemmTileTraits& traits_of_role(
                const std::array<GemmTileTraits, K_GEMM_TILE_ROLE_COUNT>& shapes,
                GemmTileRole role) noexcept
            {
                return shapes[static_cast<std::size_t>(role)];
            }
        } // namespace Detail

        /**
         * @brief Chooses the tile role of one product: the pin, else the table, else the model,
         *        always within the device's hard limits.
         * @param shapes The catalog's sizes for the product's dtype, in the order of GemmTileRole.
         * @param pin A role name, or empty for none.
         * @see modules/kernels/blas_seam_kernels/README.md
         */
        template<std::size_t N>
        [[nodiscard]] inline GemmTileChoice choose_gemm_tile(
            const Device::Profile& profile,
            const std::array<GemmTileTraits, K_GEMM_TILE_ROLE_COUNT>& shapes,
            const GemmTileProduct& product, std::string_view pin,
            const std::array<GemmTileEntry, N>& table) noexcept
        {
            GemmTileChoice choice;

            if (!pin.empty())
            {
                const std::optional<GemmTileRole> pinned = gemm_tile_role_from(pin);
                if (!pinned)
                    choice.pin = GemmTilePinStatus::Unknown;
                else if (!admits_gemm_tile(profile, Detail::traits_of_role(shapes, *pinned)))
                    choice.pin = GemmTilePinStatus::Refused;
                else
                {
                    choice.pin = GemmTilePinStatus::Applied;
                    choice.role = pinned;
                    return choice;
                }
            }

            const GemmSizeClass size_class = gemm_size_class(
                profile, Detail::traits_of_role(shapes, GemmTileRole::Balanced), product);
            for (const GemmTileEntry& entry : table)
            {
                const bool matches = entry.architecture == profile.architecture &&
                                     entry.architecture != Device::Architecture::Unknown &&
                                     entry.dtype == product.dtype &&
                                     entry.size_class == size_class;
                if (matches && admits_gemm_tile(profile, Detail::traits_of_role(shapes, entry.role)))
                {
                    choice.role = entry.role;
                    return choice;
                }
            }

            const std::int64_t units = static_cast<std::int64_t>(profile.compute_units);
            for (const GemmTileTraits& shape : shapes)
            {
                if (!admits_gemm_tile(profile, shape))
                    continue;
                if (!choice.role ||
                    Detail::models_better_tile(shape, Detail::traits_of_role(shapes, *choice.role),
                                               product, units))
                    choice.role = shape.role;
            }
            return choice;
        }

        /** @brief Chooses the tile role of one product against the measured table. */
        [[nodiscard]] inline GemmTileChoice choose_gemm_tile(
            const Device::Profile& profile,
            const std::array<GemmTileTraits, K_GEMM_TILE_ROLE_COUNT>& shapes,
            const GemmTileProduct& product, std::string_view pin) noexcept
        {
            return choose_gemm_tile(profile, shapes, product, pin, K_GEMM_TILE_TABLE);
        }

    } // namespace Kernels
} // namespace SushiBLAS
