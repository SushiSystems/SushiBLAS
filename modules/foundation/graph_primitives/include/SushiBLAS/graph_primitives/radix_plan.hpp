/**************************************************************************/
/* radix_plan.hpp                                                         */
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
 * @file radix_plan.hpp
 * @brief Declares the per-digit plan a counting sort's kernels read on the device.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>

namespace SushiBLAS
{
    namespace GraphPrimitives
    {
        namespace Detail
        {
            /** @brief Holds the number of bits one radix digit covers. */
            inline constexpr std::size_t K_RADIX_DIGIT_BITS = 8;

            /** @brief Holds the number of bins one radix digit sorts into. */
            inline constexpr std::size_t K_RADIX_DIGIT_BINS = std::size_t{1} << K_RADIX_DIGIT_BITS;

            /** @brief Holds the number of words each digit occupies in a radix plan. */
            inline constexpr std::size_t K_RADIX_PLAN_WORDS = 5;

            /** @brief Holds the offset of a digit's live tile count, zero when it is skipped. */
            inline constexpr std::size_t K_RADIX_PLAN_TILES = 0;

            /** @brief Holds the offset of a digit's live bin count, zero when it is skipped. */
            inline constexpr std::size_t K_RADIX_PLAN_BINS = 1;

            /** @brief Holds the offset of a digit's single-item count, zero when it is skipped. */
            inline constexpr std::size_t K_RADIX_PLAN_ONE = 2;

            /** @brief Holds the offset of the live element count, the first snapshot word. */
            inline constexpr std::size_t K_RADIX_PLAN_LIVE_N = 3;

            /** @brief Holds the offset of the flag that sends a placement into the indices. */
            inline constexpr std::size_t K_RADIX_PLAN_INTO_INDICES = 4;

            /**
             * @brief Counts the radix digits that cover every key below a key range.
             * @param key_range Exclusive key bound; zero and one both take one digit.
             * @return At least one, at most four for a range that fits a uint32.
             */
            [[nodiscard]] inline std::size_t radix_digit_count(std::size_t key_range) noexcept
            {
                std::size_t bits = 0;
                for (std::uint64_t v = (key_range > 1) ? key_range - 1 : 0; v != 0; v >>= 1)
                    ++bits;
                return (bits == 0) ? 1 : (bits + K_RADIX_DIGIT_BITS - 1) / K_RADIX_DIGIT_BITS;
            }

            /**
             * @brief Returns the radix digit of a key that starts at bit shift.
             * @param shift A multiple of K_RADIX_DIGIT_BITS below 32.
             */
            [[nodiscard]] inline std::uint32_t radix_digit(std::uint32_t key,
                                                           std::uint32_t shift) noexcept
            {
                return (key >> shift) & static_cast<std::uint32_t>(K_RADIX_DIGIT_BINS - 1);
            }

            /**
             * @brief Writes the radix plan of one run: which digits run and where each lands.
             * @param digit_count    Digits the sort was planned for.
             * @param live_key_range Live key range; must not exceed the planned capacity.
             * @param out            Array of at least digit_count * K_RADIX_PLAN_WORDS words.
             * @see modules/foundation/graph_primitives/README.md
             */
            inline void fill_radix_plan(std::size_t digit_count, std::size_t live_n,
                                        std::size_t live_tiles, std::size_t live_key_range,
                                        std::uint32_t* out) noexcept
            {
                const std::size_t live_digits = radix_digit_count(live_key_range);
                for (std::size_t d = 0; d < digit_count; ++d)
                {
                    const bool active = d < live_digits;
                    std::uint32_t* word = out + d * K_RADIX_PLAN_WORDS;
                    word[K_RADIX_PLAN_TILES] =
                        active ? static_cast<std::uint32_t>(live_tiles) : 0u;
                    word[K_RADIX_PLAN_BINS] =
                        active ? static_cast<std::uint32_t>(K_RADIX_DIGIT_BINS) : 0u;
                    word[K_RADIX_PLAN_ONE] = active ? 1u : 0u;
                    word[K_RADIX_PLAN_LIVE_N] = static_cast<std::uint32_t>(live_n);
                    word[K_RADIX_PLAN_INTO_INDICES] =
                        (active && (live_digits - 1 - d) % 2 == 0) ? 1u : 0u;
                }
            }

        } // namespace Detail
    } // namespace GraphPrimitives
} // namespace SushiBLAS
