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

    PackSpan BoundPack::segment(Core::DataType dtype) const
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
