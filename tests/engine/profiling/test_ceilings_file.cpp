/**************************************************************************/
/* test_ceilings_file.cpp                                                 */
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
 * @file test_ceilings_file.cpp
 * @brief Tests that the ceilings file reader accepts what the writer writes and refuses everything else.
 * @author Mustafa Garip
 * @see tests/engine/profiling/README.md
 */

#include <string>
#include <gtest/gtest.h>
#include <SushiBLAS/profiling/ceilings_file.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            using Profiling::CeilingsReadStatus;
            using Profiling::DeviceCeilings;

            /** @brief Returns ceilings a reader accepts, for a device with two dtypes. */
            DeviceCeilings valid_ceilings()
            {
                DeviceCeilings ceilings;
                ceilings.device_name = "Test Device 1";
                ceilings.backend = "level_zero";
                ceilings.driver = "1.2.3";
                ceilings.smoke_only = false;
                ceilings.method = {2, 5, 10.0, 0.05};
                ceilings.compute = {{"f32", 1.5e13, 0.01, 5}, {"f16", 3.0e13, 0.02, 5}};
                ceilings.bandwidth = {7.5e11, 0.015, 5};
                ceilings.launch_floor_ns = 5200.0;
                return ceilings;
            }

            /** @brief Replaces the first occurrence of one text with another. */
            std::string replaced(std::string text, const std::string& from, const std::string& to)
            {
                const std::size_t at = text.find(from);
                EXPECT_NE(at, std::string::npos) << from;
                if (at != std::string::npos)
                    text.replace(at, from.size(), to);
                return text;
            }
        } // namespace

        TEST(Small_CeilingsFile, TheWriterOutputPassesTheReader)
        {
            const DeviceCeilings written = valid_ceilings();

            const Profiling::CeilingsReadResult read =
                Profiling::read_ceilings(Profiling::write_ceilings(written));

            ASSERT_TRUE(read.ok()) << read.message;
            EXPECT_EQ(read.ceilings.device_name, written.device_name);
            EXPECT_EQ(read.ceilings.backend, written.backend);
            EXPECT_EQ(read.ceilings.driver, written.driver);
            EXPECT_EQ(read.ceilings.smoke_only, written.smoke_only);
            EXPECT_EQ(read.ceilings.method.runs, written.method.runs);
            EXPECT_DOUBLE_EQ(read.ceilings.method.min_span_ms, written.method.min_span_ms);
            ASSERT_EQ(read.ceilings.compute.size(), written.compute.size());
            EXPECT_EQ(read.ceilings.compute[1].dtype, "f16");
            EXPECT_DOUBLE_EQ(read.ceilings.compute[0].flops_per_second, 1.5e13);
            EXPECT_DOUBLE_EQ(read.ceilings.bandwidth.bytes_per_second, 7.5e11);
            EXPECT_DOUBLE_EQ(read.ceilings.launch_floor_ns, 5200.0);
        }

        TEST(Small_CeilingsFile, RefusesAnUnknownVersion)
        {
            const std::string text = replaced(Profiling::write_ceilings(valid_ceilings()),
                                              "\"version\": 1", "\"version\": 2");

            const Profiling::CeilingsReadResult read = Profiling::read_ceilings(text);

            EXPECT_EQ(read.status, CeilingsReadStatus::UnknownVersion);
            EXPECT_FALSE(read.message.empty());
        }

        TEST(Small_CeilingsFile, RefusesAMissingBandwidth)
        {
            const std::string text = replaced(Profiling::write_ceilings(valid_ceilings()),
                                              "\"bandwidth\"", "\"band_width\"");

            const Profiling::CeilingsReadResult read = Profiling::read_ceilings(text);

            EXPECT_EQ(read.status, CeilingsReadStatus::Invalid);
            EXPECT_NE(read.message.find("bandwidth"), std::string::npos) << read.message;
        }

        TEST(Small_CeilingsFile, RefusesTextThatIsNotJson)
        {
            EXPECT_EQ(Profiling::read_ceilings("{ not json").status, CeilingsReadStatus::NotJson);
        }

        TEST(Small_CeilingsFile, RefusesADocumentOfAnotherFormat)
        {
            const std::string text = replaced(Profiling::write_ceilings(valid_ceilings()),
                                              "sushiblas.ceilings", "sushiblas.other");

            EXPECT_EQ(Profiling::read_ceilings(text).status, CeilingsReadStatus::WrongFormat);
        }

        TEST(Small_CeilingsFile, RefusesADtypeThatIsNotMeasured)
        {
            DeviceCeilings ceilings = valid_ceilings();
            ceilings.compute[0].dtype = "f8";

            const Profiling::CeilingsReadResult read =
                Profiling::read_ceilings(Profiling::write_ceilings(ceilings));

            EXPECT_EQ(read.status, CeilingsReadStatus::Invalid);
        }

        TEST(Small_CeilingsFile, RefusesADtypeListedTwice)
        {
            DeviceCeilings ceilings = valid_ceilings();
            ceilings.compute[1].dtype = "f32";

            EXPECT_EQ(Profiling::read_ceilings(Profiling::write_ceilings(ceilings)).status,
                      CeilingsReadStatus::Invalid);
        }

        TEST(Small_CeilingsFile, RefusesACeilingThatIsNotPositive)
        {
            DeviceCeilings compute = valid_ceilings();
            compute.compute[0].flops_per_second = 0.0;
            DeviceCeilings bandwidth = valid_ceilings();
            bandwidth.bandwidth.bytes_per_second = -1.0;

            EXPECT_EQ(Profiling::read_ceilings(Profiling::write_ceilings(compute)).status,
                      CeilingsReadStatus::Invalid);
            EXPECT_EQ(Profiling::read_ceilings(Profiling::write_ceilings(bandwidth)).status,
                      CeilingsReadStatus::Invalid);
        }

        TEST(Small_CeilingsFile, RefusesAFileWithNoDtype)
        {
            DeviceCeilings ceilings = valid_ceilings();
            ceilings.compute.clear();

            EXPECT_EQ(Profiling::read_ceilings(Profiling::write_ceilings(ceilings)).status,
                      CeilingsReadStatus::Invalid);
        }

        TEST(Small_CeilingsFile, RefusesAMemberThatIsNotInTheFormat)
        {
            const std::string text = replaced(Profiling::write_ceilings(valid_ceilings()),
                                              "\"smoke_only\"", "\"smoke_only_\"");

            EXPECT_EQ(Profiling::read_ceilings(text).status, CeilingsReadStatus::Invalid);
        }

        TEST(Small_CeilingsFile, ASmokeOnlyFlagSurvivesTheRoundTrip)
        {
            DeviceCeilings ceilings = valid_ceilings();
            ceilings.smoke_only = true;

            const Profiling::CeilingsReadResult read =
                Profiling::read_ceilings(Profiling::write_ceilings(ceilings));

            ASSERT_TRUE(read.ok()) << read.message;
            EXPECT_TRUE(read.ceilings.smoke_only);
        }

        TEST(Small_CeilingsFile, TheFileNameIsTheDeviceNameInLowerCaseWithUnderscores)
        {
            EXPECT_EQ(Profiling::ceilings_file_name("NVIDIA GeForce RTX 3080 Ti"),
                      "nvidia_geforce_rtx_3080_ti.ceilings.json");
            EXPECT_EQ(Profiling::ceilings_file_name("  Intel(R) Core(TM)  "),
                      "intel_r_core_tm.ceilings.json");
            EXPECT_EQ(Profiling::ceilings_file_name(""), "device.ceilings.json");
        }
    } // namespace Test
} // namespace SushiBLAS
