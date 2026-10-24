/**************************************************************************/
/* pack_layout.cpp                                                        */
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
 * @file pack_layout.cpp
 * @brief Defines PackLayout: the segments, the offsets and the fingerprint.
 * @author Mustafa Garip
 */

#include <SushiBLAS/cluster/pack_layout.hpp>

#include <cstddef>
#include <cstdint>
#include <SushiBLAS/core/logger.hpp>

// A pack crosses the network as raw bytes, and the stack's frames are little-endian.
static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__,
              "A tensor pack holds raw little-endian elements; a big-endian host is refused.");

namespace SushiBLAS
{
    namespace
    {
        /** @brief Holds the FNV-1a 64-bit offset basis. */
        constexpr std::uint64_t K_FNV_OFFSET_BASIS = 0xcbf29ce484222325ULL;

        /** @brief Holds the FNV-1a 64-bit prime. */
        constexpr std::uint64_t K_FNV_PRIME = 0x100000001b3ULL;

        /** @brief Folds the bytes of a value into an FNV-1a hash, least significant byte first. */
        class FingerprintHash final
        {
            public:
                /** @brief Folds the low @p byte_count bytes of value, low byte first. */
                void add(std::uint64_t value, std::size_t byte_count) noexcept
                {
                    for (std::size_t index = 0; index < byte_count; ++index)
                    {
                        hash_ ^= (value >> (8 * index)) & 0xffU;
                        hash_ *= K_FNV_PRIME;
                    }
                }

                /** @brief Returns the hash of everything folded so far. */
                [[nodiscard]] std::uint64_t value() const noexcept { return hash_; }

            private:
                /// @brief The running hash.
                std::uint64_t hash_ = K_FNV_OFFSET_BASIS;
        };

        /** @brief Returns value rounded up to a multiple of K_PACK_SEGMENT_ALIGNMENT. */
        std::size_t align_up(std::size_t value)
        {
            return (value + K_PACK_SEGMENT_ALIGNMENT - 1) / K_PACK_SEGMENT_ALIGNMENT *
                   K_PACK_SEGMENT_ALIGNMENT;
        }
    } // namespace

    PackLayout PackLayout::of(SushiRuntime::span<const Tensor> tensors)
    {
        SB_THROW_IF(tensors.empty(), "PackLayout::of: a pack holds at least one tensor");

        PackLayout layout;
        layout.entries_.reserve(tensors.size());
        for (std::size_t index = 0; index < K_PACK_SEGMENT_COUNT; ++index)
            layout.segments_[index].dtype = static_cast<Core::DataType>(index);

        for (const Tensor& tensor : tensors)
        {
            const std::size_t type_index = static_cast<std::size_t>(tensor.dtype);
            SB_THROW_IF(type_index >= K_PACK_SEGMENT_COUNT,
                        "PackLayout::of: data type {} is outside the enumeration", type_index);

            PackEntry entry;
            entry.dtype = tensor.dtype;
            entry.layout = tensor.layout;
            entry.rank = tensor.rank;
            entry.shape = tensor.shape;
            entry.element_count = static_cast<std::size_t>(tensor.num_elements);
            entry.size_bytes = entry.element_count * Core::bytes_per_element(tensor.dtype);
            // offset_bytes holds the place inside the segment until the segments are placed.
            PackSegment& segment = layout.segments_[type_index];
            entry.offset_bytes = segment.size_bytes;
            segment.size_bytes += entry.size_bytes;
            segment.element_count += entry.element_count;
            layout.entries_.push_back(entry);
        }

        std::size_t next_offset = 0;
        for (PackSegment& segment : layout.segments_)
        {
            segment.offset_bytes = next_offset;
            if (segment.size_bytes != 0)
                next_offset = align_up(next_offset + segment.size_bytes);
        }
        layout.size_bytes_ = next_offset;
        SB_THROW_IF(layout.size_bytes_ == 0, "PackLayout::of: the tensors hold no element");

        for (PackEntry& entry : layout.entries_)
            entry.offset_bytes += layout.segment(entry.dtype).offset_bytes;

        FingerprintHash hash;
        hash.add(K_PACK_FINGERPRINT_VERSION, 4);
        hash.add(layout.entries_.size(), 8);
        for (const PackEntry& entry : layout.entries_)
        {
            hash.add(static_cast<std::uint64_t>(entry.dtype), 1);
            hash.add(Core::bytes_per_element(entry.dtype), 8);
            hash.add(static_cast<std::uint64_t>(entry.layout), 1);
            hash.add(static_cast<std::uint64_t>(entry.rank), 4);
            for (std::int32_t axis = 0; axis < entry.rank; ++axis)
                hash.add(static_cast<std::uint64_t>(entry.shape[axis]), 8);
            hash.add(entry.offset_bytes, 8);
            hash.add(entry.size_bytes, 8);
        }
        for (const PackSegment& segment : layout.segments_)
        {
            hash.add(segment.offset_bytes, 8);
            hash.add(segment.size_bytes, 8);
        }
        hash.add(layout.size_bytes_, 8);
        layout.fingerprint_ = hash.value();
        return layout;
    }

    const PackSegment& PackLayout::segment(Core::DataType dtype) const noexcept
    {
        return segments_[static_cast<std::size_t>(dtype) % K_PACK_SEGMENT_COUNT];
    }

    bool PackLayout::matches(const PackEntry& entry, const Tensor& tensor) noexcept
    {
        if (entry.dtype != tensor.dtype || entry.layout != tensor.layout ||
            entry.rank != tensor.rank)
            return false;

        for (std::int32_t axis = 0; axis < entry.rank; ++axis)
        {
            if (entry.shape[axis] != tensor.shape[axis])
                return false;
        }
        return true;
    }
} // namespace SushiBLAS
