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
 * @file test_ceilings_file.cpp
 * @brief Tests that the ceilings file reader accepts what the writer writes and refuses everything else.
 * @author Mustafa Garip
 * @see tests/engine/profiling/README.md
 */

#include <string>
#include <vector>
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

            /** @brief Removes the object member @p key and the comma after it. */
            std::string without_object_member(std::string text, const std::string& key)
            {
                const std::size_t at = text.find("\"" + key + "\"");
                const std::size_t close = text.find('}', at);
                const std::size_t comma = text.find(',', close);
                EXPECT_NE(at, std::string::npos) << key;
                EXPECT_NE(comma, std::string::npos) << key;
                if (at != std::string::npos && comma != std::string::npos)
                    text.erase(at, comma + 1 - at);
                return text;
            }
        } // namespace

        /** @brief Runs the round trip once with smoke_only false and once true. */
        class Small_CeilingsFileRoundTrip : public ::testing::TestWithParam<bool>
        {
        };

        TEST_P(Small_CeilingsFileRoundTrip, TheWriterOutputPassesTheReader)
        {
            DeviceCeilings written = valid_ceilings();
            written.smoke_only = GetParam();

            const Profiling::CeilingsReadResult read =
                Profiling::read_ceilings(Profiling::write_ceilings(written));

            ASSERT_TRUE(read.ok()) << read.message;
            EXPECT_EQ(read.ceilings.device_name, written.device_name);
            EXPECT_EQ(read.ceilings.backend, written.backend);
            EXPECT_EQ(read.ceilings.driver, written.driver);
            EXPECT_EQ(read.ceilings.smoke_only, GetParam());
            EXPECT_EQ(read.ceilings.method.runs, written.method.runs);
            EXPECT_DOUBLE_EQ(read.ceilings.method.min_span_ms, written.method.min_span_ms);
            ASSERT_EQ(read.ceilings.compute.size(), written.compute.size());
            EXPECT_EQ(read.ceilings.compute[1].dtype, "f16");
            EXPECT_DOUBLE_EQ(read.ceilings.compute[0].flops_per_second, 1.5e13);
            EXPECT_DOUBLE_EQ(read.ceilings.bandwidth.bytes_per_second, 7.5e11);
            EXPECT_DOUBLE_EQ(read.ceilings.launch_floor_ns, 5200.0);
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_CeilingsFileRoundTrip, ::testing::Values(false, true),
            [](const ::testing::TestParamInfo<bool>& info)
            { return std::string(info.param ? "SmokeOnly" : "Measured"); });

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
            const std::string text =
                without_object_member(Profiling::write_ceilings(valid_ceilings()), "bandwidth");

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

        namespace
        {
            /** @brief Names a ceiling set to a value the reader must refuse. */
            struct NonPositiveRow
            {
                const char* name;                  /**< The row name a failure shows. */
                void (*spoil)(DeviceCeilings&);    /**< Sets one ceiling to a value that is not positive. */
            };

            /** @brief Returns one row per ceiling. */
            std::vector<NonPositiveRow> non_positive_rows()
            {
                return {{"ZeroCompute",
                         [](DeviceCeilings& ceilings) { ceilings.compute[0].flops_per_second = 0.0; }},
                        {"NegativeBandwidth",
                         [](DeviceCeilings& ceilings) { ceilings.bandwidth.bytes_per_second = -1.0; }}};
            }
        } // namespace

        /** @brief Runs one row of non_positive_rows(). */
        class Small_CeilingsFileNonPositive : public ::testing::TestWithParam<NonPositiveRow>
        {
        };

        TEST_P(Small_CeilingsFileNonPositive, RefusesACeilingThatIsNotPositive)
        {
            DeviceCeilings ceilings = valid_ceilings();
            GetParam().spoil(ceilings);

            EXPECT_EQ(Profiling::read_ceilings(Profiling::write_ceilings(ceilings)).status,
                      CeilingsReadStatus::Invalid);
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_CeilingsFileNonPositive, ::testing::ValuesIn(non_positive_rows()),
            [](const ::testing::TestParamInfo<NonPositiveRow>& info)
            { return std::string(info.param.name); });

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

        namespace
        {
            /** @brief Names a device name and the file name derived from it. */
            struct FileNameRow
            {
                const char* name;        /**< The row name a failure shows. */
                const char* device;      /**< The device name. */
                const char* expected;    /**< The file name the rule gives. */
            };

            /** @brief Returns one row per shape of device name. */
            std::vector<FileNameRow> file_name_rows()
            {
                return {{"PunctuationAndDigits", "NVIDIA GeForce RTX 3080 Ti",
                         "nvidia_geforce_rtx_3080_ti.ceilings.json"},
                        {"TrimmedBrackets", "  Intel(R) Core(TM)  ", "intel_r_core_tm.ceilings.json"},
                        {"EmptyName", "", "device.ceilings.json"}};
            }
        } // namespace

        /** @brief Runs one row of file_name_rows(). */
        class Small_CeilingsFileName : public ::testing::TestWithParam<FileNameRow>
        {
        };

        TEST_P(Small_CeilingsFileName, TheFileNameIsTheDeviceNameInLowerCaseWithUnderscores)
        {
            EXPECT_EQ(Profiling::ceilings_file_name(GetParam().device), GetParam().expected);
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_CeilingsFileName, ::testing::ValuesIn(file_name_rows()),
            [](const ::testing::TestParamInfo<FileNameRow>& info)
            { return std::string(info.param.name); });
    } // namespace Test
} // namespace SushiBLAS
