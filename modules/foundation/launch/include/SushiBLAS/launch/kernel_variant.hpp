/**************************************************************************/
/* kernel_variant.hpp                                                     */
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
 * @file kernel_variant.hpp
 * @brief Declares the note a kernel writes when it picks a variant, and the observer that reads it.
 * @author Mustafa Garip
 * @see modules/foundation/launch/README.md
 */

#pragma once

#include <atomic>

namespace SushiBLAS
{
    namespace Device
    {
        /** @brief Names the variant a kernel launched, where it chooses among several. */
        enum class KernelVariant
        {
            GemmLocalTiled,  /**< The tiled GEMM that stages its tiles in local memory. */
            GemmJointMatrix, /**< The GEMM on the matrix units. */
            GemmDepthSplit,  /**< The GEMM that splits the depth over work groups. */
            GemmNoLocal,     /**< The GEMM that uses no local memory. */
            GemmPacked       /**< The GEMM over packed panels. */
        };

        /** @brief The function a note is handed to; it must not throw. */
        using KernelVariantObserver = void (*)(KernelVariant) noexcept;

        /** @brief Returns the slot that holds the observer; empty until one is set. */
        inline std::atomic<KernelVariantObserver>& kernel_variant_slot() noexcept
        {
            static std::atomic<KernelVariantObserver> slot{nullptr};
            return slot;
        }

        /** @brief Sets the observer of kernel variant notes; null removes it. */
        inline void set_kernel_variant_observer(KernelVariantObserver observer) noexcept
        {
            kernel_variant_slot().store(observer, std::memory_order_release);
        }

        /** @brief Tells the observer, if one is set, that a kernel launched @p variant. */
        inline void note_kernel_variant(KernelVariant variant) noexcept
        {
            const KernelVariantObserver observer =
                kernel_variant_slot().load(std::memory_order_acquire);
            if (observer != nullptr)
                observer(variant);
        }
    } // namespace Device
} // namespace SushiBLAS
