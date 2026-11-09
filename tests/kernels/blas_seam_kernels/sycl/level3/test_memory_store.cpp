/**************************************************************************/
/* test_memory_store.cpp                                                  */
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
 * @file test_memory_store.cpp
 * @brief Tests that the memory store wants the macro-blocks above its row limit and counts each write.
 * @author Mustafa Garip
 */

#include <array>
#include <cstdint>
#include <gtest/gtest.h>
#include <SushiBLAS/blas_seam_kernels/sycl/level3/memory_store.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        using Kernels::SYCL::Detail::MemoryStore;

        /** @brief A row-major C of 4 rows and 3 columns with a counter per element. */
        class Small_MemoryStore : public ::testing::Test
        {
            protected:
                static constexpr int64_t K_WIDTH = 3;

                std::array<double, 12> c{};
                std::array<unsigned, 12> calls{};
        };

        TEST_F(Small_MemoryStore, WantsABlockThatStartsBelowTheRowLimit)
        {
            const MemoryStore<double> store(K_WIDTH, calls.data(), 2);

            EXPECT_TRUE(store.wants(1, 4, 0, 3));
        }

        TEST_F(Small_MemoryStore, SkipsABlockThatStartsAtTheRowLimit)
        {
            const MemoryStore<double> store(K_WIDTH, calls.data(), 2);

            EXPECT_FALSE(store.wants(2, 4, 0, 3));
        }

        TEST_F(Small_MemoryStore, StoresASumAtItsRowMajorPosition)
        {
            const MemoryStore<double> store(K_WIDTH, calls.data(), 4);

            store(c.data(), 2, 1, 7.5);

            EXPECT_EQ(c[2 * K_WIDTH + 1], 7.5);
        }

        TEST_F(Small_MemoryStore, CountsEveryWriteOfAnElement)
        {
            const MemoryStore<double> store(K_WIDTH, calls.data(), 4);

            store(c.data(), 3, 2, 1.0);
            store(c.data(), 3, 2, 2.0);

            EXPECT_EQ(calls[3 * K_WIDTH + 2], 2u);
        }
    } // namespace Test
} // namespace SushiBLAS
