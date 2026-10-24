/**************************************************************************/
/* bound_pack.hpp                                                         */
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
 * @file bound_pack.hpp
 * @brief Declares BoundPack, the typed segments of a pack a kernel received.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <type_traits>
#include <SushiBLAS/cluster/pack_layout.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/logger.hpp>
#include <SushiBLAS/tensor/host_data_type.hpp>

namespace SushiBLAS
{
    /** @brief Names the elements of one segment, or of one tensor, of a bound pack. */
    struct PackSpan
    {
        Core::DataType dtype = Core::DataType::Float32; /**< The type of every element. */
        void* data = nullptr;                           /**< The first element; null when empty. */
        std::size_t count = 0;                          /**< The elements named. */

        /**
         * @brief Returns the elements as @p Element.
         * @throws std::runtime_error when Element is not the span's type.
         */
        template<typename Element>
        [[nodiscard]] Element* as() const
        {
            constexpr Core::DataType K_ASKED =
                Detail::HostDataType<std::remove_const_t<Element>>::K_VALUE;
            SB_THROW_IF(K_ASKED != dtype, "PackSpan::as: the span holds data type {}, not {}",
                        static_cast<int>(dtype), static_cast<int>(K_ASKED));
            return static_cast<Element*>(data);
        }
    };

    /**
     * @brief Hands back the segments and tensors of a pack whose memory was checked.
     *
     * It owns nothing: the memory is the view's, valid as long as the view is.
     */
    class BoundPack final
    {
        public:
            /** @brief Binds a base address to a layout; bind_views() checks the size first. */
            BoundPack(void* base, const PackLayout& layout) noexcept;

            /**
             * @brief Returns the segment of a data type; empty when no tensor has it.
             * @throws std::runtime_error for a value outside Core::DataType.
             */
            [[nodiscard]] PackSpan segment(Core::DataType dtype) const;

            /**
             * @brief Returns the elements of the tensor at a position of the layout.
             * @pre index is below the layout's entry count.
             */
            [[nodiscard]] PackSpan tensor(std::size_t index) const;

        private:
            /// @brief The pack's first byte.
            std::byte* base_;

            /// @brief Where each tensor lies; outlives this object.
            const PackLayout* layout_;
    };
} // namespace SushiBLAS
