/**************************************************************************/
/* gemm_tile_catalog.hpp                                                  */
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
 * @file gemm_tile_catalog.hpp
 * @brief Declares the tile shapes the tiled GEMM is compiled for and their stable role names.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>
#include <SushiBLAS/blas_seam_kernels/level3/gemm_tile_shape.hpp>

namespace SushiBLAS
{
    namespace Kernels
    {
        /**
         * @brief Names the role of one compiled tile shape; the sizes of a role differ per dtype.
         * @see modules/kernels/blas_seam_kernels/README.md
         */
        enum class GemmTileRole : std::uint8_t
        {
            Balanced, /**< The default shape: an 8 x 8 work-group and a 16-deep K tile. */
            Wide,     /**< Twice the Balanced block rows and a 32-deep K tile, for large products. */
            Compact   /**< Half the Balanced block on each axis, for products that do not fill the device. */
        };

        /** @brief Holds how many roles the catalog compiles. */
        inline constexpr std::size_t K_GEMM_TILE_ROLE_COUNT = 3;

        /** @brief Holds every role, in the order of GemmTileRole. */
        inline constexpr std::array<GemmTileRole, K_GEMM_TILE_ROLE_COUNT> K_GEMM_TILE_ROLES = {
            GemmTileRole::Balanced, GemmTileRole::Wide, GemmTileRole::Compact};

        /** @brief Returns the stable name of a role, the one `SB_GEMM_TILE` and the table use. */
        [[nodiscard]] inline constexpr std::string_view to_string(GemmTileRole role) noexcept
        {
            switch (role)
            {
                case GemmTileRole::Balanced:
                    return "Balanced";
                case GemmTileRole::Wide:
                    return "Wide";
                case GemmTileRole::Compact:
                    return "Compact";
            }
            return "Unknown";
        }

        /** @brief Returns the role a stable name names, or nothing for a name outside the catalog. */
        [[nodiscard]] inline constexpr std::optional<GemmTileRole>
        gemm_tile_role_from(std::string_view name) noexcept
        {
            for (const GemmTileRole role : K_GEMM_TILE_ROLES)
                if (to_string(role) == name)
                    return role;
            return std::nullopt;
        }

        /**
         * @brief Names the GemmTileShape one role compiles to for element type T.
         * @see modules/kernels/blas_seam_kernels/README.md
         */
        template<typename T, GemmTileRole Role>
        struct GemmTileOf;

        /** @brief Compiles Balanced: 8 x 8 work-items, 256 accumulator bytes, a 16-deep K tile. */
        template<typename T>
        struct GemmTileOf<T, GemmTileRole::Balanced>
        {
            /** @brief The shape. */
            using type = GemmTileShape<T, 8, 8, 256, 16>;
        };

        /** @brief Compiles Wide: 16 x 8 work-items, 256 accumulator bytes, a 32-deep K tile. */
        template<typename T>
        struct GemmTileOf<T, GemmTileRole::Wide>
        {
            /** @brief The shape. */
            using type = GemmTileShape<T, 16, 8, 256, 32>;
        };

        /** @brief Compiles Compact: 8 x 8 work-items, 64 accumulator bytes, a 16-deep K tile. */
        template<typename T>
        struct GemmTileOf<T, GemmTileRole::Compact>
        {
            /** @brief The shape. */
            using type = GemmTileShape<T, 8, 8, 64, 16>;
        };

        /** @brief Shorthand for `GemmTileOf<T, Role>::type`. */
        template<typename T, GemmTileRole Role>
        using GemmTile = typename GemmTileOf<T, Role>::type;

        /**
         * @brief Holds the sizes of one compiled shape as plain values, for the host chooser.
         * @see modules/kernels/blas_seam_kernels/README.md
         */
        struct GemmTileTraits
        {
            /** @brief The role the sizes belong to. */
            GemmTileRole role = GemmTileRole::Balanced;

            /** @brief Rows of C one work-group covers. */
            std::int64_t block_m = 1;

            /** @brief Columns of C one work-group covers. */
            std::int64_t block_n = 1;

            /** @brief Work-items per work-group. */
            std::size_t work_group_items = 1;

            /** @brief Bytes of local memory one work-group allocates. */
            std::size_t local_bytes = 0;
        };

        /** @brief Returns the plain sizes of the shape one role compiles to for T. */
        template<typename T, GemmTileRole Role>
        [[nodiscard]] inline constexpr GemmTileTraits gemm_tile_traits_of() noexcept
        {
            using Shape = GemmTile<T, Role>;
            return GemmTileTraits{Role, Shape::K_BLOCK_M, Shape::K_BLOCK_N,
                                  static_cast<std::size_t>(Shape::K_WG_ITEMS),
                                  Shape::local_mem_bytes()};
        }

        /** @brief Returns the plain sizes of every catalog shape for T, in the order of GemmTileRole. */
        template<typename T>
        [[nodiscard]] inline constexpr std::array<GemmTileTraits, K_GEMM_TILE_ROLE_COUNT>
        gemm_tile_traits() noexcept
        {
            return {gemm_tile_traits_of<T, GemmTileRole::Balanced>(),
                    gemm_tile_traits_of<T, GemmTileRole::Wide>(),
                    gemm_tile_traits_of<T, GemmTileRole::Compact>()};
        }

    } // namespace Kernels
} // namespace SushiBLAS
