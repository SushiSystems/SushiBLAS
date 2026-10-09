/**************************************************************************/
/* rng_stream.hpp                                                         */
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
 * @file rng_stream.hpp
 * @brief Declares RngStream, the seed and counter offset random operations draw from.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <mutex>

namespace SushiBLAS
{
    /// @brief Holds the seed and first offset one random task draws from.
    struct RngPosition
    {
        std::uint64_t seed;   /**< Seed of the stream at reservation time. */
        std::uint64_t offset; /**< First of the offsets reserved for the task. */
    };

    /**
     * @brief Holds one counter-based random stream: a seed and an offset counter under one lock.
     * @see modules/engine/engine/README.md
     */
    class RngStream final
    {
        public:
            RngStream() = default;
            RngStream(const RngStream&) = delete;
            RngStream& operator=(const RngStream&) = delete;
            RngStream(RngStream&&) = delete;
            RngStream& operator=(RngStream&&) = delete;
            ~RngStream() = default;

            /// @brief Sets the seed and rewinds the offset counter to zero, as one atomic step.
            void reseed(std::uint64_t seed)
            {
                const std::lock_guard<std::mutex> lock(mutex_);
                seed_ = seed;
                offset_ = 0;
            }

            /// @brief Returns the current seed.
            [[nodiscard]] std::uint64_t seed() const
            {
                const std::lock_guard<std::mutex> lock(mutex_);
                return seed_;
            }

            /**
             * @brief Reserves @p count consecutive offsets and returns them with their seed.
             * @param count Number of offsets taken; concurrent callers receive disjoint ranges.
             */
            [[nodiscard]] RngPosition reserve(std::uint64_t count)
            {
                const std::lock_guard<std::mutex> lock(mutex_);
                const RngPosition position{seed_, offset_};
                offset_ += count;
                return position;
            }

        private:
            /// @brief Guards seed_ and offset_ so a reservation never pairs two seeds' state.
            mutable std::mutex mutex_;

            /// @brief Seed every reserved offset is paired with.
            std::uint64_t seed_ = 0;

            /// @brief Next offset reserve() hands out.
            std::uint64_t offset_ = 0;
    };
} // namespace SushiBLAS
