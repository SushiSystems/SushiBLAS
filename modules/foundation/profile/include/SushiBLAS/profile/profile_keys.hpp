/**************************************************************************/
/* profile_keys.hpp                                                       */
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
 * @file profile_keys.hpp
 * @brief Declares the group and fact names SushiBLAS writes into a record.
 * @author Mustafa Garip
 * @see modules/foundation/profile/README.md
 */

#pragma once

#include <array>
#include <string_view>

namespace SushiBLAS
{
    namespace Internal::Profile
    {
        /** @brief Names the group every SushiBLAS zone is charged to. */
        inline constexpr std::string_view K_GROUP_NAME = "BLAS";

        /** @brief Names the fact that holds the SushiBLAS version. */
        inline constexpr std::string_view K_FACT_VERSION = "sushiblas.version";

        /** @brief Names the fact that holds the execution mode recorded tasks default to. */
        inline constexpr std::string_view K_FACT_DEFAULT_MODE = "sushiblas.default_mode";

        /** @brief Names the fact that holds the value of the GEMM tile pin, or `unpinned`. */
        inline constexpr std::string_view K_FACT_TILE_PIN = "sushiblas.gemm_tile_pin";

        /** @brief Names the fact that holds the compiled profiling level of the library. */
        inline constexpr std::string_view K_FACT_PROFILE_LEVEL = "sushiblas.profile_level";

        /** @brief Opens the name of the fact that states what became of a device's ceilings; the track name follows. */
        inline constexpr std::string_view K_FACT_CEILINGS_PREFIX = "sushiblas.ceilings.";

        /** @brief Lists every name above once, for the check that none repeats. */
        inline constexpr std::array<std::string_view, 6> K_ALL_KEYS{
            K_GROUP_NAME,
            K_FACT_VERSION,
            K_FACT_DEFAULT_MODE,
            K_FACT_TILE_PIN,
            K_FACT_PROFILE_LEVEL,
            K_FACT_CEILINGS_PREFIX};
    } // namespace Internal::Profile
} // namespace SushiBLAS
