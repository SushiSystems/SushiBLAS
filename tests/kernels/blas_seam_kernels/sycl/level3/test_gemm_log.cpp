/**************************************************************************/
/* test_gemm_log.cpp                                                      */
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
 * @file test_gemm_log.cpp
 * @brief Tests that the GEMM log writes each fallback and applied pin once, to the sink it holds.
 * @author Mustafa Garip
 */

#include <string>
#include <vector>
#include <cstddef>
#include <gtest/gtest.h>
#include <SushiBLAS/launch/device_policy.hpp>
#include <SushiBLAS/blas_seam_kernels/level3/gemm_tile_choice.hpp>
#include <SushiBLAS/blas_seam_kernels/sycl/level3/gemm_log.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        using Kernels::SYCL::Detail::GemmFallbackLog;
        using Kernels::SYCL::Detail::GemmLogSink;

        /** @brief Keeps the text of each record it takes. */
        class RecordingGemmLogSink final : public GemmLogSink
        {
            public:
                /** @brief Keeps a WARN. */
                void warn(const std::string& message) override
                {
                    warnings.push_back(message);
                }

                /** @brief Keeps an INFO. */
                void info(const std::string& message) override
                {
                    infos.push_back(message);
                }

                std::vector<std::string> warnings; /**< The WARN records, in order. */
                std::vector<std::string> infos;    /**< The INFO records, in order. */
        };

        /** @brief Builds a choice that applied a pin, or one the given status describes. */
        static Kernels::GemmTileChoice choice_with(Kernels::GemmTilePinStatus status)
        {
            Kernels::GemmTileChoice choice;
            choice.pin = status;
            choice.role = Kernels::GemmTileRole::Balanced;

            return choice;
        }

        /** @brief Holds a log over a recording sink. */
        class Small_GemmFallbackLog : public ::testing::Test
        {
            protected:
                RecordingGemmLogSink sink;
                GemmFallbackLog log{sink};
        };

        TEST_F(Small_GemmFallbackLog, WarnsOncePerDeviceAndTile)
        {
            const Device::Profile first_device;

            log.warn_no_local_fallback(first_device, 65536, 256);
            log.warn_no_local_fallback(first_device, 65536, 256);
            log.warn_no_local_fallback(first_device, 65536, 256);

            EXPECT_EQ(sink.warnings.size(), std::size_t{1});
        }

        TEST_F(Small_GemmFallbackLog, WarnsAgainForAnotherDeviceOrAnotherTile)
        {
            const Device::Profile first_device;
            const Device::Profile second_device;

            log.warn_no_local_fallback(first_device, 65536, 256);
            log.warn_no_local_fallback(second_device, 65536, 256);
            log.warn_no_local_fallback(first_device, 131072, 256);

            EXPECT_EQ(sink.warnings.size(), std::size_t{3});
        }

        TEST_F(Small_GemmFallbackLog, TheWarningNamesTheTileAndTheFallback)
        {
            const Device::Profile device;

            log.warn_no_local_fallback(device, 65536, 256);

            ASSERT_EQ(sink.warnings.size(), std::size_t{1});
            EXPECT_NE(sink.warnings.front().find("falling back to the no-local variant"),
                      std::string::npos);
            EXPECT_NE(sink.warnings.front().find("65536 B and 256 work-items"), std::string::npos);
        }

        TEST_F(Small_GemmFallbackLog, AnAppliedPinIsStatedOncePerLog)
        {
            const Kernels::GemmTileChoice choice = choice_with(Kernels::GemmTilePinStatus::Applied);

            log.state_applied_gemm_tile_pin(choice, 64, 64, 64);
            log.state_applied_gemm_tile_pin(choice, 128, 128, 128);

            ASSERT_EQ(sink.infos.size(), std::size_t{1});
            EXPECT_NE(sink.infos.front().find("selects the"), std::string::npos);
            EXPECT_NE(sink.infos.front().find("first seen for 64x64x64"), std::string::npos);
        }

        TEST_F(Small_GemmFallbackLog, APinThatWasNotAppliedStatesNothing)
        {
            const Kernels::GemmTileChoice choice = choice_with(Kernels::GemmTilePinStatus::Unknown);

            log.state_applied_gemm_tile_pin(choice, 64, 64, 64);

            EXPECT_EQ(sink.infos.size(), std::size_t{0});
        }

        TEST_F(Small_GemmFallbackLog, AUnknownPinIsWarnedOncePerLog)
        {
            log.warn_rejected_gemm_tile_pin(Kernels::GemmTilePinStatus::Unknown);
            log.warn_rejected_gemm_tile_pin(Kernels::GemmTilePinStatus::Unknown);

            ASSERT_EQ(sink.warnings.size(), std::size_t{1});
            EXPECT_NE(sink.warnings.front().find("names no catalog shape"), std::string::npos);
            EXPECT_NE(sink.warnings.front().find(Device::K_GEMM_TILE_PIN_VARIABLE),
                      std::string::npos);
        }

        TEST_F(Small_GemmFallbackLog, ARefusedPinNamesTheHardLimit)
        {
            log.warn_rejected_gemm_tile_pin(Kernels::GemmTilePinStatus::Refused);

            ASSERT_EQ(sink.warnings.size(), std::size_t{1});
            EXPECT_NE(sink.warnings.front().find("is past a hard limit of the device"),
                      std::string::npos);
        }

        TEST_F(Small_GemmFallbackLog, AnAppliedOrAbsentPinWarnsNothing)
        {
            log.warn_rejected_gemm_tile_pin(Kernels::GemmTilePinStatus::Applied);
            log.warn_rejected_gemm_tile_pin(Kernels::GemmTilePinStatus::None);

            EXPECT_EQ(sink.warnings.size(), std::size_t{0});
        }
    } // namespace Test
} // namespace SushiBLAS
