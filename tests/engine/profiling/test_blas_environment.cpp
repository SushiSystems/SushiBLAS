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
 * @file test_blas_environment.cpp
 * @brief Tests the facts BlasEnvironment states: SushiBLAS's own, and the ceilings of each device, read through an in-memory source.
 * @author Mustafa Garip
 * @see tests/engine/profiling/README.md
 */

#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/core/version.hpp>
#include <SushiBLAS/profiling/blas_environment.hpp>
#include <SushiBLAS/profiling/ceilings_file.hpp>
#include <SushiBLAS/profiling/in_memory_ceilings_source.hpp>
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

            /** @brief Holds the ceilings files of one test in memory, so no folder is needed. */
            class Small_BlasEnvironment : public ::testing::Test
            {
                protected:
                    /** @brief Puts valid ceilings for a device into the source. */
                    void put_ceilings_for(const std::string& device_name, bool smoke_only)
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
                        source_->put(Profiling::ceilings_file_name(device_name),
                                     Profiling::write_ceilings(ceilings));
                    }

                    /** @brief Returns an environment over the source for the given devices. */
                    BlasEnvironment environment_of(std::vector<DeviceFacts> devices) const
                    {
                        return BlasEnvironment(std::move(devices), source_);
                    }

                    /// @brief The ceilings files of one test.
                    std::shared_ptr<Profiling::InMemoryCeilingsSource> source_ =
                        std::make_shared<Profiling::InMemoryCeilingsSource>();
            };
        } // namespace

        TEST_F(Small_BlasEnvironment, StatesTheTilePinAndTheDefaultMode)
        {
            const BlasEnvironment environment = environment_of({});

            const SushiProfiler::RecordFacts facts = facts_of(environment);

            EXPECT_EQ(value_of(facts.environment, "sushiblas.default_mode"), "auto");
            EXPECT_NE(value_of(facts.environment, "sushiblas.gemm_tile_pin"), "<absent>");
        }

        TEST_F(Small_BlasEnvironment, StatesTheVersionAndTheProfileLevelAsBuildFacts)
        {
            const BlasEnvironment environment = environment_of({});

            const SushiProfiler::RecordFacts facts = facts_of(environment);

            EXPECT_EQ(value_of(facts.build, "sushiblas.version"), SUSHIBLAS_VERSION_STRING);
            EXPECT_NE(value_of(facts.build, "sushiblas.profile_level"), "<absent>");
        }

        TEST_F(Small_BlasEnvironment, StatesTheCeilingsOfADeviceWithAFile)
        {
            put_ceilings_for("Device A", false);
            const BlasEnvironment environment = environment_of({DeviceFacts{"node/0/Device A", "Device A"}});

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

        TEST_F(Small_BlasEnvironment, MarksTheCeilingsOfASmokeOnlyDevice)
        {
            put_ceilings_for("Device A", true);
            const BlasEnvironment environment = environment_of({DeviceFacts{"node/0/Device A", "Device A"}});

            const SushiProfiler::RecordFacts facts = facts_of(environment);

            EXPECT_EQ(value_of(facts.environment, "sushiblas.ceilings.node/0/Device A"),
                      "smoke_only");
        }

        TEST_F(Small_BlasEnvironment, StatesUnmeasuredWithoutAFile)
        {
            const BlasEnvironment environment = environment_of({DeviceFacts{"node/0/Device B", "Device B"}});

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

        TEST_F(Small_BlasEnvironment, StatesRefusedForAFileTheReaderRefuses)
        {
            source_->put(Profiling::ceilings_file_name("Device C"),
                         "{ \"format\": \"sushiblas.ceilings\", \"version\": 9 }");
            const BlasEnvironment environment = environment_of({DeviceFacts{"node/0/Device C", "Device C"}});

            const SushiProfiler::RecordFacts facts = facts_of(environment);

            EXPECT_EQ(value_of(facts.environment, "sushiblas.ceilings.node/0/Device C"),
                      "refused");
            EXPECT_EQ(value_of(facts.environment,
                               SushiProfiler::DeviceSpanKeys::peak_bytes_per_s_fact(
                                   "node/0/Device C")),
                      "<absent>");
        }

        TEST_F(Small_BlasEnvironment, EachDeviceGetsItsOwnCeilings)
        {
            put_ceilings_for("Device A", false);
            const BlasEnvironment environment = environment_of({DeviceFacts{"node/0/Device A", "Device A"},
                                                                DeviceFacts{"node/1/Device B", "Device B"}});

            const SushiProfiler::RecordFacts facts = facts_of(environment);

            EXPECT_EQ(value_of(facts.environment, "sushiblas.ceilings.node/0/Device A"),
                      "measured");
            EXPECT_EQ(value_of(facts.environment, "sushiblas.ceilings.node/1/Device B"),
                      "unmeasured");
        }
    } // namespace Test
} // namespace SushiBLAS
