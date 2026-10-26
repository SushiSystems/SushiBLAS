/**************************************************************************/
/* test_kernel_variant.cpp                                                */
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
 * @file test_kernel_variant.cpp
 * @brief Tests that a kernel variant note reaches the observer that is set, and only that one.
 * @author Mustafa Garip
 * @see tests/foundation/launch/README.md
 */

#include <gtest/gtest.h>
#include <SushiBLAS/launch/kernel_variant.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief The last variant the test observer saw.
            Device::KernelVariant g_seen = Device::KernelVariant::GemmNoLocal;

            /// @brief How many notes the test observer saw.
            int g_count = 0;

            /// @brief Records the note it is handed.
            void remember(Device::KernelVariant variant) noexcept
            {
                g_seen = variant;
                ++g_count;
            }

            /// @brief Removes the observer when a test ends.
            class Small_KernelVariant : public ::testing::Test
            {
                protected:
                    void SetUp() override
                    {
                        g_count = 0;
                        g_seen = Device::KernelVariant::GemmNoLocal;
                    }

                    void TearDown() override
                    {
                        Device::set_kernel_variant_observer(nullptr);
                    }
            };
        } // namespace

        TEST_F(Small_KernelVariant, ANoteWithNoObserverDoesNothing)
        {
            Device::set_kernel_variant_observer(nullptr);
            Device::note_kernel_variant(Device::KernelVariant::GemmPacked);

            EXPECT_EQ(g_count, 0);
        }

        TEST_F(Small_KernelVariant, ANoteReachesTheObserverWithItsVariant)
        {
            Device::set_kernel_variant_observer(&remember);
            Device::note_kernel_variant(Device::KernelVariant::GemmDepthSplit);

            EXPECT_EQ(g_count, 1);
            EXPECT_EQ(g_seen, Device::KernelVariant::GemmDepthSplit);
        }

        TEST_F(Small_KernelVariant, ARemovedObserverStopsReceivingNotes)
        {
            Device::set_kernel_variant_observer(&remember);
            Device::set_kernel_variant_observer(nullptr);
            Device::note_kernel_variant(Device::KernelVariant::GemmLocalTiled);

            EXPECT_EQ(g_count, 0);
        }
    } // namespace Test
} // namespace SushiBLAS
