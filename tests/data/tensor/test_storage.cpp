/**************************************************************************/
/* test_storage.cpp                                                       */
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
 * @file test_storage.cpp
 * @brief Tests that Storage rounds its size up, aligns its allocation and refuses a bad request.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiBLAS/tensor/storage.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief Returns a storage of n_bytes on the shared runtime's first device.
        SushiRuntime::sushi_ptr<Storage> storage_of(std::size_t n_bytes)
        {
            return SushiRuntime::make_sushi<Storage>(TestEnvironment::runtime(), n_bytes);
        }

        TEST(Unit_Storage, RoundsItsSizeUpToTheAlignment)
        {
            EXPECT_EQ(storage_of(1)->size_bytes(), K_STORAGE_ALIGNMENT);
            EXPECT_EQ(storage_of(K_STORAGE_ALIGNMENT)->size_bytes(), K_STORAGE_ALIGNMENT);
            EXPECT_EQ(storage_of(K_STORAGE_ALIGNMENT + 1)->size_bytes(), 2 * K_STORAGE_ALIGNMENT);
        }

        TEST(Unit_Storage, ItsAllocationIsAligned)
        {
            const auto storage = storage_of(1000);

            EXPECT_EQ(reinterpret_cast<std::uintptr_t>(storage->allocation_base()) %
                          K_STORAGE_ALIGNMENT,
                      std::uintptr_t{0});
        }

        TEST(Unit_Storage, ReportsItsResidencyDeviceAndQueue)
        {
            const auto storage = storage_of(64);

            EXPECT_EQ(storage->residency(), SushiRuntime::Residency::Shared);
            EXPECT_EQ(storage->device_index(), std::size_t{0});
            EXPECT_EQ(&storage->queue(), &TestEnvironment::queue(0));
        }

        TEST(Unit_Storage, RefusesAnEmptyRequestAndADeviceItDoesNotHave)
        {
            EXPECT_THROW((void)storage_of(0), std::runtime_error);
            EXPECT_THROW((void)SushiRuntime::make_sushi<Storage>(
                             TestEnvironment::runtime(), 64, SushiRuntime::Residency::Shared,
                             TestEnvironment::device_count()),
                         std::runtime_error);
        }

        TEST(Unit_Storage, ReturnsItsAllocationWhenTheLastHandleGoes)
        {
            const std::size_t before = TestEnvironment::live_allocation_count();
            {
                const auto storage = storage_of(256);
                EXPECT_EQ(TestEnvironment::live_allocation_count(), before + 1);
            }

            EXPECT_EQ(TestEnvironment::live_allocation_count(), before);
        }
    } // namespace Test
} // namespace SushiBLAS
