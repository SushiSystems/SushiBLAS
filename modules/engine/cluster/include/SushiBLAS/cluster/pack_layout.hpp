/**************************************************************************/
/* pack_layout.hpp                                                        */
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
 * @file pack_layout.hpp
 * @brief Declares PackLayout, where each tensor of a pack lies in its flat buffer.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>

namespace SushiBLAS
{
    /** @brief Holds the alignment of every segment of a pack, in bytes. */
    inline constexpr std::size_t K_PACK_SEGMENT_ALIGNMENT = 128;

    /** @brief Holds how many segments a pack has: one per data type. */
    inline constexpr std::size_t K_PACK_SEGMENT_COUNT = 7;

    /** @brief Holds the version of the byte sequence a fingerprint hashes. */
    inline constexpr std::uint32_t K_PACK_FINGERPRINT_VERSION = 1;

    /** @brief Describes one tensor of a pack: its type, layout, shape and place. */
    struct PackEntry
    {
        Core::DataType dtype = Core::DataType::Float32;           /**< The element type. */
        Core::Layout layout = Core::Layout::RowMajor;             /**< The element order. */
        std::int32_t rank = 0;                                    /**< The number of axes. */
        std::array<std::int64_t, Core::K_MAX_TENSOR_RANK> shape{}; /**< The extents. */
        std::size_t element_count = 0;                            /**< The elements held. */
        std::size_t offset_bytes = 0;                             /**< The start in the pack. */
        std::size_t size_bytes = 0;                               /**< The bytes it takes. */
    };

    /** @brief Describes the segment of one data type: where it starts and what it holds. */
    struct PackSegment
    {
        Core::DataType dtype = Core::DataType::Float32; /**< The type of every element. */
        std::size_t offset_bytes = 0;                   /**< The start, a multiple of 128. */
        std::size_t size_bytes = 0;                     /**< The bytes of its tensors. */
        std::size_t element_count = 0;                  /**< The elements of its tensors. */
    };

    /**
     * @brief Lays an ordered list of tensors out in one flat buffer, one segment per type.
     *
     * Segments follow the order of Core::DataType; inside a segment the tensors keep the
     * order they were given in. The fingerprint hashes every entry, segment and type size.
     * @see modules/engine/cluster/README.md
     */
    class PackLayout final
    {
        public:
            /**
             * @brief Lays out tensors of the given types, layouts and shapes.
             * @throws std::runtime_error when the list is empty or holds no element.
             */
            [[nodiscard]] static PackLayout of(SushiRuntime::span<const Tensor> tensors);

            /** @brief Returns the tensors, in the order they were given. */
            [[nodiscard]] const std::vector<PackEntry>& entries() const noexcept
            {
                return entries_;
            }

            /** @brief Returns the segment of a data type; it is empty when no tensor has it. */
            [[nodiscard]] const PackSegment& segment(Core::DataType dtype) const noexcept;

            /** @brief Returns the bytes up to the end of the last segment. */
            [[nodiscard]] std::size_t size_bytes() const noexcept { return size_bytes_; }

            /** @brief Returns the 64-bit FNV-1a hash of the layout's canonical bytes. */
            [[nodiscard]] std::uint64_t fingerprint() const noexcept { return fingerprint_; }

            /** @brief Returns whether a tensor matches an entry in type, layout and shape. */
            [[nodiscard]] static bool matches(const PackEntry& entry,
                                              const Tensor& tensor) noexcept;

        private:
            PackLayout() = default;

            /// @brief The tensors, in the order they were given.
            std::vector<PackEntry> entries_;

            /// @brief The segment of each type, indexed by the value of Core::DataType.
            std::array<PackSegment, K_PACK_SEGMENT_COUNT> segments_{};

            /// @brief The end of the last segment.
            std::size_t size_bytes_ = 0;

            /// @brief The hash of entries_, segments_ and size_bytes_.
            std::uint64_t fingerprint_ = 0;
    };
} // namespace SushiBLAS
