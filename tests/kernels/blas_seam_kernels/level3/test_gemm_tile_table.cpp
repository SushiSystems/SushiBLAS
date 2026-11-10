/**************************************************************************/
/* test_gemm_tile_table.cpp                                               */
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
 * @file test_gemm_tile_table.cpp
 * @brief Tests that the GEMM tile chooser matches a table entry only by a known architecture, and the key rule of a table.
 * @author Mustafa Garip
 */

#include <array>
#include <cstdint>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/launch/device_policy.hpp>
#include <SushiBLAS/blas_seam_kernels/level3/gemm_tile_catalog.hpp>
#include <SushiBLAS/blas_seam_kernels/level3/gemm_tile_choice.hpp>
#include <SushiBLAS/blas_seam_kernels/level3/gemm_tile_table.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief Returns the profile of a GPU whose architecture is not known. */
            [[nodiscard]] Device::Profile unknown_gpu()
            {
                Device::Profile p;
                p.is_cpu = false;
                p.has_dedicated_local_memory = true;
                p.architecture = Device::Architecture::Unknown;
                p.compute_units = 16;
                p.local_mem_bytes = 32 * 1024;
                p.local_mem_per_compute_unit = 32 * 1024;
                p.max_work_group_size = 1024;
                p.sub_group_size = 16;
                return p;
            }

            /** @brief Holds a table to inject and whether its keys are all distinct. */
            struct KeyRow
            {
                std::string name;
                std::vector<Kernels::GemmTileEntry> table;
                bool unique;
            };

            /** @brief Returns an entry of the given key whose role is the given one. */
            [[nodiscard]] Kernels::GemmTileEntry entry(Device::Architecture architecture, Core::DataType dtype,
                                                       Kernels::GemmSizeClass size_class,
                                                       Kernels::GemmTileRole role)
            {
                return Kernels::GemmTileEntry{architecture, dtype, size_class, role};
            }

            /** @brief Names a row by its description. */
            std::string key_row_name(const ::testing::TestParamInfo<KeyRow>& info)
            {
                return info.param.name;
            }
        } // namespace

        /** @brief Runs the key rule over each injected table. */
        class Small_GemmTileTableKeys : public ::testing::TestWithParam<KeyRow>
        {
        };

        TEST_P(Small_GemmTileTableKeys, ATableHoldsOneAnswerPerKeyExactlyWhenNoTwoEntriesShareOne)
        {
            EXPECT_EQ(Kernels::has_unique_keys(GetParam().table), GetParam().unique);
        }

        // The key is the architecture, the dtype and the size class; the role is the answer and never part of it.
        INSTANTIATE_TEST_SUITE_P(
            , Small_GemmTileTableKeys,
            ::testing::Values(
                KeyRow{"AnEmptyTable", {}, true},
                KeyRow{"OneEntry",
                       {entry(Device::Architecture::Unknown, Core::DataType::Float32,
                              Kernels::GemmSizeClass::Large, Kernels::GemmTileRole::Compact)},
                       true},
                KeyRow{"TheSameKeyTwice",
                       {entry(Device::Architecture::Unknown, Core::DataType::Float32,
                              Kernels::GemmSizeClass::Large, Kernels::GemmTileRole::Compact),
                        entry(Device::Architecture::Unknown, Core::DataType::Float32,
                              Kernels::GemmSizeClass::Large, Kernels::GemmTileRole::Compact)},
                       false},
                KeyRow{"TheSameKeyWithAnotherRole",
                       {entry(Device::Architecture::Unknown, Core::DataType::Float32,
                              Kernels::GemmSizeClass::Large, Kernels::GemmTileRole::Compact),
                        entry(Device::Architecture::Unknown, Core::DataType::Float32,
                              Kernels::GemmSizeClass::Large, Kernels::GemmTileRole::Balanced)},
                       false},
                KeyRow{"TheSameKeyAtTheEndOfThree",
                       {entry(Device::Architecture::Unknown, Core::DataType::Float32,
                              Kernels::GemmSizeClass::Small, Kernels::GemmTileRole::Compact),
                        entry(Device::Architecture::Unknown, Core::DataType::Float64,
                              Kernels::GemmSizeClass::Small, Kernels::GemmTileRole::Compact),
                        entry(Device::Architecture::Unknown, Core::DataType::Float32,
                              Kernels::GemmSizeClass::Small, Kernels::GemmTileRole::Balanced)},
                       false},
                KeyRow{"KeysThatDifferInTheSizeClass",
                       {entry(Device::Architecture::Unknown, Core::DataType::Float32,
                              Kernels::GemmSizeClass::Small, Kernels::GemmTileRole::Compact),
                        entry(Device::Architecture::Unknown, Core::DataType::Float32,
                              Kernels::GemmSizeClass::Large, Kernels::GemmTileRole::Compact)},
                       true},
                KeyRow{"KeysThatDifferInTheDtype",
                       {entry(Device::Architecture::Unknown, Core::DataType::Float32,
                              Kernels::GemmSizeClass::Large, Kernels::GemmTileRole::Compact),
                        entry(Device::Architecture::Unknown, Core::DataType::Float64,
                              Kernels::GemmSizeClass::Large, Kernels::GemmTileRole::Compact)},
                       true}),
            key_row_name);

        TEST(Small_GemmTileTable, AnEntryNamingAnUnknownArchitectureNeverMatchesAnUnknownDevice)
        {
            const Kernels::GemmTileProduct product{2048, 2048, 2048, 1, Core::DataType::Float32};
            const std::array<Kernels::GemmTileEntry, 0> empty{};
            const std::array<Kernels::GemmTileEntry, 1> table = {Kernels::GemmTileEntry{
                Device::Architecture::Unknown, Core::DataType::Float32,
                Kernels::GemmSizeClass::Large, Kernels::GemmTileRole::Compact}};

            const Kernels::GemmTileChoice modeled = Kernels::choose_gemm_tile(
                unknown_gpu(), Kernels::gemm_tile_traits<float>(), product, "", empty);
            const Kernels::GemmTileChoice tabled = Kernels::choose_gemm_tile(
                unknown_gpu(), Kernels::gemm_tile_traits<float>(), product, "", table);

            EXPECT_NE(modeled.role, Kernels::GemmTileRole::Compact);
            EXPECT_EQ(tabled.role, modeled.role);
        }
    } // namespace Test
} // namespace SushiBLAS
