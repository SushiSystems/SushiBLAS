/**************************************************************************/
/* tensor_pack.hpp                                                        */
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
 * @file tensor_pack.hpp
 * @brief Declares TensorPack, a flat shared buffer an ordered list of tensors is gathered into.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <SushiBLAS/cluster/pack_layout.hpp>
#include <SushiBLAS/engine/task_handle.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiBLAS/tensor/storage.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiRuntime/core/sushi_ptr.hpp>
#include <SushiRuntime/distributed/buffer_view.hpp>

namespace SushiBLAS
{
    /**
     * @brief Owns the buffer of one pack layout and records the copies into and out of it.
     *
     * The buffer is shared USM on device 0, so Cluster::share() accepts it. Copies of a
     * pack name the same buffer.
     * @see modules/engine/cluster/README.md
     */
    class TensorPack final
    {
        public:
            /** @brief Allocates the buffer of a layout on a runtime. */
            TensorPack(SushiRuntime::Runtime& runtime, PackLayout layout);

            /** @brief Returns where each tensor lies in the buffer. */
            [[nodiscard]] const PackLayout& layout() const noexcept { return layout_; }

            /** @brief Returns the layout's fingerprint. */
            [[nodiscard]] std::uint64_t fingerprint() const noexcept
            {
                return layout_.fingerprint();
            }

            /** @brief Returns the buffer's address. */
            [[nodiscard]] void* data() const noexcept { return storage_->allocation_base(); }

            /** @brief Returns the bytes of the layout, which the buffer holds at least. */
            [[nodiscard]] std::size_t size_bytes() const noexcept { return layout_.size_bytes(); }

            /** @brief Returns the view a registered task names the buffer by. */
            [[nodiscard]] SushiRuntime::Distributed::BufferView view() const noexcept
            {
                return SushiRuntime::Distributed::BufferView{data(), size_bytes()};
            }

            /**
             * @brief Records one task that copies each tensor into its place in the buffer.
             * @param sources One contiguous tensor per entry, in the layout's order.
             * @throws std::runtime_error when a tensor does not match its entry.
             */
            TaskHandle record_gather(TaskRecorder& recorder,
                                     SushiRuntime::span<const Tensor> sources) const;

            /**
             * @brief Records one task that copies each place of the buffer into its tensor.
             * @param targets One contiguous tensor per entry, in the layout's order.
             * @throws std::runtime_error when a tensor does not match its entry.
             */
            TaskHandle record_scatter(TaskRecorder& recorder,
                                      SushiRuntime::span<const Tensor> targets) const;

        private:
            /** @brief Returns a Float32 tensor over the whole buffer, the pack as an operand. */
            [[nodiscard]] Tensor as_operand() const;

            /// @brief Where each tensor lies.
            PackLayout layout_;

            /// @brief The shared USM the tensors are packed into.
            SushiRuntime::sushi_ptr<Storage> storage_;
    };
} // namespace SushiBLAS
