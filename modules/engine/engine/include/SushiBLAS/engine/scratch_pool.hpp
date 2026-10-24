/**************************************************************************/
/* scratch_pool.hpp                                                       */
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
 * @file scratch_pool.hpp
 * @brief Declares ScratchPool, the owner of the device scratch blocks an Engine lends.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <SushiBLAS/engine/scratch_shelf.hpp>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiRuntime/core/sushi_ptr.hpp>

namespace SushiBLAS
{
    /**
     * @class ScratchPool
     * @brief Owns the shelf of device scratch blocks that ScratchBuffer leases come from.
     * @see modules/engine/engine/README.md
     */
    class ScratchPool final
    {
        public:
            /**
             * @brief Builds an empty pool over a runtime.
             * @param runtime Allocates every block; must outlive every ScratchBuffer leased here.
             */
            explicit ScratchPool(SushiRuntime::Runtime& runtime)
                : shelf_(SushiRuntime::make_sushi<Detail::ScratchShelf>(runtime))
            {

            }

            ScratchPool(const ScratchPool&) = delete;
            ScratchPool& operator=(const ScratchPool&) = delete;
            ScratchPool(ScratchPool&&) = delete;
            ScratchPool& operator=(ScratchPool&&) = delete;
            ~ScratchPool() = default;

            /** @brief Returns the shelf ScratchBuffer takes blocks from; never null. */
            [[nodiscard]] const SushiRuntime::sushi_ptr<Detail::ScratchShelf>& shelf() const noexcept
            {
                return shelf_;
            }

            /** @brief Returns the runtime the pool was built over. */
            [[nodiscard]] SushiRuntime::Runtime& runtime() const noexcept
            {
                return shelf_->runtime();
            }

            /** @brief Returns how many blocks sit idle in the pool. */
            [[nodiscard]] std::size_t idle_block_count() const { return shelf_->idle_block_count(); }

            /** @brief Frees every idle block; blocks out on lease are not touched. */
            void trim() { shelf_->trim(); }

        private:
            /// @brief The shared shelf; lent blocks hold it too, so it outlives the pool if they do.
            SushiRuntime::sushi_ptr<Detail::ScratchShelf> shelf_;
    };
} // namespace SushiBLAS
