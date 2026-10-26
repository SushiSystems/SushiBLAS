/**************************************************************************/
/* gemm_tile_pin.hpp                                                      */
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
 * @file gemm_tile_pin.hpp
 * @brief Declares the read of the environment variable that pins a GEMM tile shape by name.
 * @author Mustafa Garip
 */

#pragma once

#include <string>
#include <cstdlib>

namespace SushiBLAS
{
    namespace Device
    {
        /**
         * @brief Names the environment variable that pins the GEMM tile shape.
         * @see modules/foundation/launch/README.md
         */
        inline constexpr const char* K_GEMM_TILE_PIN_VARIABLE = "SB_GEMM_TILE";

        /**
         * @brief Returns the pin a variable's value holds.
         * @param value The value getenv returned, or nullptr when the variable is unset.
         * @return The value, or an empty string for nullptr.
         * @see modules/foundation/launch/README.md
         */
        [[nodiscard]] inline std::string gemm_tile_pin_from(const char* value)
        {
            return value == nullptr ? std::string() : std::string(value);
        }

        /**
         * @brief Returns the pin of this process, read from the environment on first use.
         * @return A reference valid for the rest of the process; empty when no pin is set.
         * @see modules/foundation/launch/README.md
         */
        [[nodiscard]] inline const std::string& gemm_tile_pin()
        {
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
            static const std::string pin =
                gemm_tile_pin_from(std::getenv(K_GEMM_TILE_PIN_VARIABLE));
#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif
            return pin;
        }
    } // namespace Device
} // namespace SushiBLAS
