/**************************************************************************/
/* storage_contract.hpp                                                   */
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
 * @file storage_contract.hpp
 * @brief Declares the checks every Storage adapter passes, run by one typed suite per adapter.
 * @author Mustafa Garip
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/tensor/storage.hpp>
#include <SushiBLAS/tensor/tensor.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief The byte the contract writes over a storage.
        inline constexpr unsigned char K_STORAGE_CONTRACT_PATTERN = 0x5A;

        /** @brief Holds the checks that define a Storage; an adapter's suite calls each one. */
        struct StorageContract
        {
            /** @brief The size is the request rounded up to K_STORAGE_ALIGNMENT. */
            template<typename Subject>
            static void expect_size_rounds_up_to_the_alignment(const Subject& subject)
            {
                EXPECT_EQ(subject.storage_of(1)->size_bytes(), K_STORAGE_ALIGNMENT);
                EXPECT_EQ(subject.storage_of(K_STORAGE_ALIGNMENT)->size_bytes(), K_STORAGE_ALIGNMENT);
                EXPECT_EQ(subject.storage_of(K_STORAGE_ALIGNMENT + 1)->size_bytes(),
                          2 * K_STORAGE_ALIGNMENT);
            }

            /** @brief The base of the allocation is a multiple of K_STORAGE_ALIGNMENT. */
            template<typename Subject>
            static void expect_base_is_aligned(const Subject& subject)
            {
                const auto storage = subject.storage_of(1000);

                EXPECT_EQ(reinterpret_cast<std::uintptr_t>(storage->allocation_base()) %
                              K_STORAGE_ALIGNMENT,
                          std::uintptr_t{0});
            }

            /** @brief Every byte of the allocation can be written and read back. */
            template<typename Subject>
            static void expect_allocation_holds_what_is_written(const Subject& subject)
            {
                const auto storage = subject.storage_of(1000);
                const std::size_t size = storage->size_bytes();

                std::memset(storage->allocation_base(), K_STORAGE_CONTRACT_PATTERN, size);

                const auto* bytes = static_cast<const unsigned char*>(storage->allocation_base());
                EXPECT_EQ(std::vector<unsigned char>(bytes, bytes + size),
                          std::vector<unsigned char>(size, K_STORAGE_CONTRACT_PATTERN));
            }

            /** @brief A tensor over the storage addresses its first element at the base. */
            template<typename Subject>
            static void expect_tensor_data_is_the_base(const Subject& subject)
            {
                const auto storage = subject.storage_of(64);
                const Tensor tensor(storage, {4, 4}, Core::DataType::Float32);

                EXPECT_EQ(tensor.data(), storage->allocation_base());
            }

            /** @brief The storage reports the residency and device its adapter places it in. */
            template<typename Subject>
            static void expect_reports_its_residency_and_device(const Subject& subject)
            {
                const auto storage = subject.storage_of(64);

                EXPECT_EQ(storage->residency(), Subject::residency());
                EXPECT_EQ(storage->device_index(), std::size_t{0});
            }

            /** @brief A request of zero bytes is refused. */
            template<typename Subject>
            static void expect_empty_request_is_refused(const Subject& subject)
            {
                EXPECT_THROW((void)subject.storage_of(0), std::runtime_error);
            }

            /** @brief A request whose rounded size overflows is refused. */
            template<typename Subject>
            static void expect_overflowing_request_is_refused(const Subject& subject)
            {
                EXPECT_THROW((void)subject.storage_of(SIZE_MAX), std::runtime_error);
            }

            /** @brief Two live storages share no byte. */
            template<typename Subject>
            static void expect_live_storages_do_not_overlap(const Subject& subject)
            {
                const auto first = subject.storage_of(256);
                const auto second = subject.storage_of(256);

                const auto low = std::min(reinterpret_cast<std::uintptr_t>(first->allocation_base()),
                                          reinterpret_cast<std::uintptr_t>(second->allocation_base()));
                const auto high = std::max(reinterpret_cast<std::uintptr_t>(first->allocation_base()),
                                           reinterpret_cast<std::uintptr_t>(second->allocation_base()));
                EXPECT_GE(high - low, first->size_bytes());
            }
        };

    } // namespace Test
} // namespace SushiBLAS
