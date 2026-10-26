/**************************************************************************/
/* test_blas_environment.cpp                                              */
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
 * @file test_blas_environment.cpp
 * @brief Tests the facts BlasEnvironment states: SushiBLAS's own, and the ceilings of each device.
 * @author Mustafa Garip
 * @see tests/engine/profiling/README.md
 */

#include <string>
#include <vector>
#include <fstream>
#include <filesystem>
#include <gtest/gtest.h>
#include <SushiBLAS/core/version.hpp>
#include <SushiBLAS/profiling/blas_environment.hpp>
#include <SushiBLAS/profiling/ceilings_file.hpp>
#include <SushiProfiler/environment/record_facts.hpp>
#include <SushiProfiler/record/device_span_keys.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            using Profiling::BlasEnvironment;
            using Profiling::DeviceFacts;

            /** @brief Returns the value of a fact, or a marker when the fact is not there. */
            std::string value_of(const std::vector<SushiProfiler::Fact>& facts,
                                 const std::string& name)
            {
                for (const SushiProfiler::Fact& fact : facts)
                {
                    if (fact.name == name)
                        return fact.value;
                }
                return "<absent>";
            }

            /** @brief Returns facts gathered from one environment. */
            SushiProfiler::RecordFacts facts_of(const BlasEnvironment& environment)
            {
                return SushiProfiler::gather_facts({&environment});
            }

            /** @brief Owns a folder under the temporary directory and removes it afterwards. */
            class Medium_BlasEnvironment : public ::testing::Test
            {
                protected:
                    void SetUp() override
                    {
                        const std::string name = std::string("sushiblas_profiling_") +
                            ::testing::UnitTest::GetInstance()->current_test_info()->name();
                        folder_ = std::filesystem::temp_directory_path() / name;
                        std::filesystem::remove_all(folder_);
                        std::filesystem::create_directories(folder_);
                    }

                    void TearDown() override
                    {
                        std::filesystem::remove_all(folder_);
                    }

                    /** @brief Writes valid ceilings for a device into the folder. */
                    void write_ceilings_for(const std::string& device_name, bool smoke_only)
                    {
                        Profiling::DeviceCeilings ceilings;
                        ceilings.device_name = device_name;
                        ceilings.backend = "test";
                        ceilings.driver = "1";
                        ceilings.smoke_only = smoke_only;
                        ceilings.method = {2, 5, 10.0, 0.05};
                        ceilings.compute = {{"f32", 2.0e12, 0.01, 5}, {"f16", 4.0e12, 0.01, 5}};
                        ceilings.bandwidth = {5.0e11, 0.01, 5};
                        ceilings.launch_floor_ns = 4000.0;
                        std::ofstream file(folder_ / Profiling::ceilings_file_name(device_name),
                                           std::ios::binary);
                        file << Profiling::write_ceilings(ceilings);
                    }

                    /// @brief The folder the ceilings files of one test live in.
                    std::filesystem::path folder_;
            };
        } // namespace

        TEST_F(Medium_BlasEnvironment, StatesTheTilePinAndTheDefaultMode)
        {
            const BlasEnvironment environment({}, folder_.string());

            const SushiProfiler::RecordFacts facts = facts_of(environment);

            EXPECT_EQ(value_of(facts.environment, "sushiblas.default_mode"), "auto");
            EXPECT_NE(value_of(facts.environment, "sushiblas.gemm_tile_pin"), "<absent>");
        }

        TEST_F(Medium_BlasEnvironment, StatesTheVersionAndTheProfileLevelAsBuildFacts)
        {
            const BlasEnvironment environment({}, folder_.string());

            const SushiProfiler::RecordFacts facts = facts_of(environment);

            EXPECT_EQ(value_of(facts.build, "sushiblas.version"), SUSHIBLAS_VERSION_STRING);
            EXPECT_NE(value_of(facts.build, "sushiblas.profile_level"), "<absent>");
        }

        TEST_F(Medium_BlasEnvironment, StatesTheCeilingsOfADeviceWithAFile)
        {
            write_ceilings_for("Device A", false);
            const BlasEnvironment environment({DeviceFacts{"node/0/Device A", "Device A"}},
                                              folder_.string());

            const SushiProfiler::RecordFacts facts = facts_of(environment);

            const std::string track = "node/0/Device A";
            EXPECT_EQ(value_of(facts.environment,
                               SushiProfiler::DeviceSpanKeys::peak_flops_fact(track, "f32")),
                      "2000000000000");
            EXPECT_EQ(value_of(facts.environment,
                               SushiProfiler::DeviceSpanKeys::peak_flops_fact(track, "f16")),
                      "4000000000000");
            EXPECT_EQ(value_of(facts.environment,
                               SushiProfiler::DeviceSpanKeys::peak_bytes_per_s_fact(track)),
                      "500000000000");
            EXPECT_EQ(value_of(facts.environment, "sushiblas.ceilings." + track), "measured");
        }

        TEST_F(Medium_BlasEnvironment, MarksTheCeilingsOfASmokeOnlyDevice)
        {
            write_ceilings_for("Device A", true);
            const BlasEnvironment environment({DeviceFacts{"node/0/Device A", "Device A"}},
                                              folder_.string());

            const SushiProfiler::RecordFacts facts = facts_of(environment);

            EXPECT_EQ(value_of(facts.environment, "sushiblas.ceilings.node/0/Device A"),
                      "smoke_only");
        }

        TEST_F(Medium_BlasEnvironment, StatesUnmeasuredWithoutAFile)
        {
            const BlasEnvironment environment({DeviceFacts{"node/0/Device B", "Device B"}},
                                              folder_.string());

            const SushiProfiler::RecordFacts facts = facts_of(environment);

            const std::string track = "node/0/Device B";
            EXPECT_EQ(value_of(facts.environment, "sushiblas.ceilings." + track), "unmeasured");
            EXPECT_EQ(value_of(facts.environment,
                               SushiProfiler::DeviceSpanKeys::peak_bytes_per_s_fact(track)),
                      "<absent>");
            EXPECT_EQ(value_of(facts.environment,
                               SushiProfiler::DeviceSpanKeys::peak_flops_fact(track, "f32")),
                      "<absent>");
        }

        TEST_F(Medium_BlasEnvironment, StatesRefusedForAFileTheReaderRefuses)
        {
            std::ofstream file(folder_ / Profiling::ceilings_file_name("Device C"),
                               std::ios::binary);
            file << "{ \"format\": \"sushiblas.ceilings\", \"version\": 9 }";
            file.close();
            const BlasEnvironment environment({DeviceFacts{"node/0/Device C", "Device C"}},
                                              folder_.string());

            const SushiProfiler::RecordFacts facts = facts_of(environment);

            EXPECT_EQ(value_of(facts.environment, "sushiblas.ceilings.node/0/Device C"),
                      "refused");
            EXPECT_EQ(value_of(facts.environment,
                               SushiProfiler::DeviceSpanKeys::peak_bytes_per_s_fact(
                                   "node/0/Device C")),
                      "<absent>");
        }

        TEST_F(Medium_BlasEnvironment, EachDeviceGetsItsOwnCeilings)
        {
            write_ceilings_for("Device A", false);
            const BlasEnvironment environment({DeviceFacts{"node/0/Device A", "Device A"},
                                               DeviceFacts{"node/1/Device B", "Device B"}},
                                              folder_.string());

            const SushiProfiler::RecordFacts facts = facts_of(environment);

            EXPECT_EQ(value_of(facts.environment, "sushiblas.ceilings.node/0/Device A"),
                      "measured");
            EXPECT_EQ(value_of(facts.environment, "sushiblas.ceilings.node/1/Device B"),
                      "unmeasured");
        }
    } // namespace Test
} // namespace SushiBLAS
