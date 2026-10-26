/**************************************************************************/
/* test_profile_hooks.cpp                                                 */
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
 * @file test_profile_hooks.cpp
 * @brief Tests that the profiling hooks run their body once, under their lock.
 * @author Mustafa Garip
 * @see tests/engine/engine/README.md
 */

#include <mutex>
#include <stdexcept>
#include <thread>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/profile_hooks.hpp>
#include <SushiBLAS/launch/kernel_variant.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Small_ProfileHooks, ARecordingBodyRunsOnceWhileTheMutexIsHeld)
        {
            std::mutex mutex;
            int runs = 0;
            bool others_blocked = false;

            Internal::in_recording(mutex, [&]
            {
                ++runs;
                std::thread other([&] { others_blocked = !mutex.try_lock(); });
                other.join();
            });

            EXPECT_EQ(runs, 1);
            EXPECT_TRUE(others_blocked);
            EXPECT_TRUE(mutex.try_lock());
            mutex.unlock();
        }

        TEST(Small_ProfileHooks, AnExceptionOfARecordingBodyLeavesTheMutexFree)
        {
            std::mutex mutex;

            EXPECT_THROW(Internal::in_recording(mutex, [] { throw std::runtime_error("refused"); }),
                         std::runtime_error);
            EXPECT_TRUE(mutex.try_lock());
            mutex.unlock();
        }

        TEST(Small_ProfileHooks, AScratchLeaseBodyRunsOnce)
        {
            int runs = 0;

            Internal::in_scratch_lease([&] { ++runs; });

            EXPECT_EQ(runs, 1);
        }

        TEST(Small_ProfileHooks, EveryLeaseKindIsCountedWithoutACollector)
        {
            Internal::count_scratch_lease(Internal::ScratchLeaseKind::PoolHit);
            Internal::count_scratch_lease(Internal::ScratchLeaseKind::PoolMiss);
            Internal::count_scratch_lease(Internal::ScratchLeaseKind::Unpooled);
            SUCCEED();
        }

        TEST(Small_ProfileHooks, InstallingTheHooksSetsTheVariantObserverAndIsRepeatable)
        {
            Internal::install_profile_hooks();
            Internal::install_profile_hooks();

            EXPECT_NE(Device::kernel_variant_slot().load(), nullptr);
            Device::note_kernel_variant(Device::KernelVariant::GemmPacked);
        }
    } // namespace Test
} // namespace SushiBLAS
