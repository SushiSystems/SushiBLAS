/**************************************************************************/
/* bound_pack.cpp                                                         */
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
 * @file bound_pack.cpp
 * @brief Defines BoundPack: the segments and tensors of a checked pack.
 * @author Mustafa Garip
 */

#include <SushiBLAS/cluster/bound_pack.hpp>

#include <cstddef>

namespace SushiBLAS
{
    namespace
    {
        /** @brief Returns the span of bytes at an offset of a base, null when empty. */
        PackSpan span_at(std::byte* base, Core::DataType dtype, std::size_t offset_bytes,
                         std::size_t count) noexcept
        {
            PackSpan span;
            span.dtype = dtype;
            span.count = count;
            span.data = count == 0 ? nullptr : base + offset_bytes;
            return span;
        }
    } // namespace

    BoundPack::BoundPack(void* base, const PackLayout& layout) noexcept
        : base_(static_cast<std::byte*>(base)), layout_(&layout)
    {

    }

    PackSpan BoundPack::segment(Core::DataType dtype) const noexcept
    {
        const PackSegment& segment = layout_->segment(dtype);
        return span_at(base_, segment.dtype, segment.offset_bytes, segment.element_count);
    }

    PackSpan BoundPack::tensor(std::size_t index) const
    {
        const PackEntry& entry = layout_->entries().at(index);
        return span_at(base_, entry.dtype, entry.offset_bytes, entry.element_count);
    }
} // namespace SushiBLAS
