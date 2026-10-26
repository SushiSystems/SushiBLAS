/**************************************************************************/
/* test_gemm_tile_choice.cpp                                              */
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
 * @file test_gemm_tile_choice.cpp
 * @brief Tests the tile chooser of the tiled GEMM on fake device profiles.
 * @author Mustafa Garip
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/launch/device_policy.hpp>
#include <SushiBLAS/blas_seam_kernels/level3/gemm_tile_catalog.hpp>
#include <SushiBLAS/blas_seam_kernels/level3/gemm_tile_choice.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief Returns a GPU profile with the given traits. */
            [[nodiscard]] Device::Profile gpu_profile(Device::Architecture architecture,
                                                      std::size_t compute_units,
                                                      std::size_t local_kib,
                                                      std::size_t sub_group_size)
            {
                Device::Profile p;
                p.is_cpu = false;
                p.has_dedicated_local_memory = true;
                p.architecture = architecture;
                p.compute_units = compute_units;
                p.local_mem_bytes = local_kib * 1024;
                p.local_mem_per_compute_unit = local_kib * 1024;
                p.max_work_group_size = 1024;
                p.sub_group_size = sub_group_size;
                return p;
            }

            /** @brief Returns the profile of an RTX 3080 Ti. */
            [[nodiscard]] Device::Profile sm_86()
            {
                Device::Profile p = gpu_profile(Device::Architecture::NvidiaSm86, 80, 48, 32);
                p.local_mem_per_compute_unit = 100 * 1024;
                return p;
            }

            /** @brief Returns the profile of an MI250X compute die. */
            [[nodiscard]] Device::Profile gfx90a()
            {
                return gpu_profile(Device::Architecture::AmdGfx90a, 110, 64, 64);
            }

            /** @brief Returns the profile of a Max 1550 stack. */
            [[nodiscard]] Device::Profile pvc()
            {
                return gpu_profile(Device::Architecture::IntelPVC, 64, 128, 32);
            }

            /** @brief Returns the profile of a GPU no table names. */
            [[nodiscard]] Device::Profile unknown_gpu()
            {
                return gpu_profile(Device::Architecture::Unknown, 16, 32, 16);
            }

            /** @brief Returns the profile of a CPU device, which has no dedicated local memory. */
            [[nodiscard]] Device::Profile cpu()
            {
                Device::Profile p;
                p.is_cpu = true;
                p.local_mem_bytes = 256 * 1024;
                p.max_work_group_size = 8192;
                p.compute_units = 12;
                return p;
            }

            /** @brief Returns a square FLOAT32 product of one batch. */
            [[nodiscard]] Kernels::GemmTileProduct square(std::int64_t order)
            {
                return Kernels::GemmTileProduct{order, order, order, 1, Core::DataType::Float32};
            }

            /** @brief Returns the chooser's role for a product, against an empty table and no pin. */
            [[nodiscard]] Kernels::GemmTileRole modeled(const Device::Profile& profile,
                                                        const Kernels::GemmTileProduct& product)
            {
                const Kernels::GemmTileChoice choice = Kernels::choose_gemm_tile(
                    profile, Kernels::gemm_tile_traits<float>(), product, "",
                    std::array<Kernels::GemmTileEntry, 0>{});
                EXPECT_TRUE(choice.role.has_value());
                return choice.role.value_or(Kernels::GemmTileRole::Balanced);
            }
        } // namespace

        TEST(Small_GemmTileChoice, ALargeProductTakesTheDensestShapeOnEveryGPU)
        {
            for (const Device::Profile& profile : {sm_86(), gfx90a(), pvc(), unknown_gpu()})
                EXPECT_EQ(modeled(profile, square(2048)), Kernels::GemmTileRole::Wide);
        }

        TEST(Small_GemmTileChoice, ASmallProductTakesTheShapeThatSpreadsItWidest)
        {
            for (const Device::Profile& profile : {sm_86(), gfx90a(), pvc(), unknown_gpu()})
                EXPECT_EQ(modeled(profile, square(128)), Kernels::GemmTileRole::Compact);
        }

        TEST(Small_GemmTileChoice, AProductOnlyBalancedFillsTakesBalanced)
        {
            // 512 x 640 gives sm_86 80 Balanced tiles and 40 Wide ones against 80 units.
            const Kernels::GemmTileProduct product{512, 640, 512, 1, Core::DataType::Float32};

            EXPECT_EQ(modeled(sm_86(), product), Kernels::GemmTileRole::Balanced);
        }

        TEST(Small_GemmTileChoice, TheSizeClassCountsBalancedTiles)
        {
            const auto balanced = Kernels::gemm_tile_traits<float>()[0];

            EXPECT_EQ(Kernels::gemm_size_class(sm_86(), balanced, square(512)),
                      Kernels::GemmSizeClass::Small);
            EXPECT_EQ(Kernels::gemm_size_class(sm_86(), balanced, square(1024)),
                      Kernels::GemmSizeClass::Large);
        }

        TEST(Small_GemmTileChoice, ACPUDeviceGetsNoTiledShape)
        {
            const Kernels::GemmTileChoice choice = Kernels::choose_gemm_tile(
                cpu(), Kernels::gemm_tile_traits<float>(), square(512), "");

            EXPECT_FALSE(choice.role.has_value());
        }

        TEST(Small_GemmTileChoice, AShapePastAHardLimitIsNeverReturned)
        {
            Device::Profile narrow = sm_86();
            narrow.max_work_group_size = 64;
            Device::Profile small_local = sm_86();
            small_local.local_mem_bytes = 16 * 1024;

            for (const Device::Profile& profile : {narrow, small_local})
                EXPECT_NE(modeled(profile, square(2048)), Kernels::GemmTileRole::Wide);
        }

        TEST(Small_GemmTileChoice, AWorkGroupOfPartialSubGroupsIsNeverReturned)
        {
            // Only Wide's 128 work-items are whole sub-groups of 128.
            Device::Profile profile = gfx90a();
            profile.sub_group_size = 128;

            EXPECT_EQ(modeled(profile, square(128)), Kernels::GemmTileRole::Wide);
        }

        TEST(Small_GemmTileChoice, LocalMemoryPerComputeUnitIsAHardLimit)
        {
            Device::Profile profile = sm_86();
            profile.local_mem_per_compute_unit = 16 * 1024;

            EXPECT_NE(modeled(profile, square(2048)), Kernels::GemmTileRole::Wide);
        }

        TEST(Small_GemmTileChoice, APinInTheCatalogIsApplied)
        {
            const Kernels::GemmTileChoice choice = Kernels::choose_gemm_tile(
                sm_86(), Kernels::gemm_tile_traits<float>(), square(2048), "Compact");

            EXPECT_EQ(choice.pin, Kernels::GemmTilePinStatus::Applied);
            EXPECT_EQ(choice.role, Kernels::GemmTileRole::Compact);
        }

        TEST(Small_GemmTileChoice, AnUnknownPinIsReportedAndTheModelChooses)
        {
            const Kernels::GemmTileChoice choice = Kernels::choose_gemm_tile(
                sm_86(), Kernels::gemm_tile_traits<float>(), square(2048), "Tile64x64x16");

            EXPECT_EQ(choice.pin, Kernels::GemmTilePinStatus::Unknown);
            EXPECT_EQ(choice.role, Kernels::GemmTileRole::Wide);
        }

        TEST(Small_GemmTileChoice, APinPastAHardLimitIsRefusedAndTheModelChooses)
        {
            Device::Profile profile = sm_86();
            profile.max_work_group_size = 64;
            const Kernels::GemmTileChoice choice = Kernels::choose_gemm_tile(
                profile, Kernels::gemm_tile_traits<float>(), square(2048), "Wide");

            EXPECT_EQ(choice.pin, Kernels::GemmTilePinStatus::Refused);
            EXPECT_EQ(choice.role, Kernels::GemmTileRole::Balanced);
        }

        TEST(Small_GemmTileChoice, AMeasuredEntryWinsOverTheModel)
        {
            const std::array<Kernels::GemmTileEntry, 1> table = {Kernels::GemmTileEntry{
                Device::Architecture::NvidiaSm86, Core::DataType::Float32,
                Kernels::GemmSizeClass::Large, Kernels::GemmTileRole::Balanced}};

            const Kernels::GemmTileChoice choice = Kernels::choose_gemm_tile(
                sm_86(), Kernels::gemm_tile_traits<float>(), square(2048), "", table);

            EXPECT_EQ(choice.role, Kernels::GemmTileRole::Balanced);
        }

        TEST(Small_GemmTileChoice, AnEntryOfAnotherArchitectureDtypeOrClassIsIgnored)
        {
            const std::array<Kernels::GemmTileEntry, 3> table = {
                Kernels::GemmTileEntry{Device::Architecture::AmdGfx90a, Core::DataType::Float32,
                                       Kernels::GemmSizeClass::Large,
                                       Kernels::GemmTileRole::Balanced},
                Kernels::GemmTileEntry{Device::Architecture::NvidiaSm86, Core::DataType::Float64,
                                       Kernels::GemmSizeClass::Large,
                                       Kernels::GemmTileRole::Balanced},
                Kernels::GemmTileEntry{Device::Architecture::NvidiaSm86, Core::DataType::Float32,
                                       Kernels::GemmSizeClass::Small,
                                       Kernels::GemmTileRole::Balanced}};

            const Kernels::GemmTileChoice choice = Kernels::choose_gemm_tile(
                sm_86(), Kernels::gemm_tile_traits<float>(), square(2048), "", table);

            EXPECT_EQ(choice.role, Kernels::GemmTileRole::Wide);
        }

        TEST(Small_GemmTileChoice, AnEntryPastAHardLimitFallsBackToTheModel)
        {
            Device::Profile profile = sm_86();
            profile.max_work_group_size = 64;
            const std::array<Kernels::GemmTileEntry, 1> table = {Kernels::GemmTileEntry{
                Device::Architecture::NvidiaSm86, Core::DataType::Float32,
                Kernels::GemmSizeClass::Large, Kernels::GemmTileRole::Wide}};

            const Kernels::GemmTileChoice choice = Kernels::choose_gemm_tile(
                profile, Kernels::gemm_tile_traits<float>(), square(2048), "", table);

            EXPECT_EQ(choice.role, Kernels::GemmTileRole::Balanced);
        }
    } // namespace Test
} // namespace SushiBLAS
