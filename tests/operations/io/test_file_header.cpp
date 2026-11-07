/**************************************************************************/
/* test_file_header.cpp                                                   */
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

/**
 * @file test_file_header.cpp
 * @brief Tests the .npy and .sushi header readers and checks on byte buffers and text, with no file and no tensor.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/io/file_header.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief Names the file the refusals mention. */
            const std::string K_PATH = "header.bin";

            /** @brief Holds a header dictionary numpy writes for a 2 x 3 float32 array. */
            const std::string K_VALID_DICT = "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }";

            /** @brief Returns the bytes of a .npy prefix: the magic, the version and a length field. */
            std::vector<unsigned char> npy_prefix(unsigned char major, unsigned char minor,
                                                  std::size_t header_bytes)
            {
                std::vector<unsigned char> bytes{0x93, 'N', 'U', 'M', 'P', 'Y', major, minor};
                const std::size_t field = major == 1 ? 2 : 4;
                for (std::size_t i = 0; i < field; ++i)
                    bytes.push_back(static_cast<unsigned char>((header_bytes >> (8 * i)) & 0xFF));
                return bytes;
            }

            /** @brief Returns the status the call refused with, or Ok when it did not refuse. */
            template<typename Call>
            IOStatus refusal_of(Call call)
            {
                try
                {
                    call();
                }
                catch (const HeaderRefusal& refusal)
                {
                    return refusal.status;
                }
                return IOStatus::Ok;
            }

            /** @brief Returns the 2 x 3 float32 row-major expectation. */
            NpyExpectation expected_npy()
            {
                NpyExpectation expected;
                expected.descr = "<f4";
                expected.column_major = false;
                expected.shape = {2, 3};
                return expected;
            }

            /** @brief Holds a prefix, the bytes of it a file has, and the status the parse gives. */
            struct PrefixRow
            {
                const char* name;
                std::vector<unsigned char> bytes;
                std::size_t size;
                IOStatus status;
            };

            /** @brief Returns the prefixes the parser refuses. */
            std::vector<PrefixRow> refused_prefix_rows()
            {
                std::vector<unsigned char> wrong_magic = npy_prefix(1, 0, 118);
                wrong_magic[1] = 'X';
                return {{"VersionFour", npy_prefix(4, 0, 118), 12, IOStatus::MalformedFile},
                        {"VersionOneOne", npy_prefix(1, 1, 118), 12, IOStatus::MalformedFile},
                        {"VersionZero", npy_prefix(0, 0, 118), 12, IOStatus::MalformedFile},
                        {"WrongMagic", wrong_magic, 12, IOStatus::MalformedFile},
                        {"EmptyFile", npy_prefix(1, 0, 118), 0, IOStatus::Truncated},
                        {"EndsInTheMagic", npy_prefix(1, 0, 118), 5, IOStatus::Truncated},
                        {"EndsInTheVersion", npy_prefix(1, 0, 118), 7, IOStatus::Truncated},
                        {"EndsInTheLengthOfVersionOne", npy_prefix(1, 0, 118), 9, IOStatus::Truncated},
                        {"EndsInTheLengthOfVersionTwo", npy_prefix(2, 0, 118), 11, IOStatus::Truncated}};
            }

            /** @brief Holds a header dictionary text. */
            struct DictRow
            {
                const char* name;
                std::string text;
            };

            /** @brief Returns the dictionaries the parser refuses as malformed. */
            std::vector<DictRow> malformed_dict_rows()
            {
                return {{"WithoutAShape", "{'descr': '<f4', 'fortran_order': False, 'shapf': (2, 3), }"},
                        {"WithoutCommas", "{'descr': '<f4' 'fortran_order': False 'shape': (2, 3) }"},
                        {"WithADuplicateKey",
                         "{'descr': '<f4', 'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }"},
                        {"WithoutAnOrder", "{'descr': '<f4', 'shape': (2, 3), }"},
                        {"WithAnUnquotedKey", "{descr: '<f4', 'fortran_order': False, 'shape': (2, 3), }"},
                        {"WithAnUnterminatedString", "{'descr': '<f4, 'fortran_order': False, 'shape': (2, 3), }"},
                        {"WithANegativeDimension", "{'descr': '<f4', 'fortran_order': False, 'shape': (-2, 3), }"},
                        {"WithADimensionBeyondInt64",
                         "{'descr': '<f4', 'fortran_order': False, 'shape': (99999999999999999999, 3), }"},
                        {"WithANonBooleanOrder", "{'descr': '<f4', 'fortran_order': 1, 'shape': (2, 3), }"},
                        {"WithoutABrace", "'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }"},
                        {"Empty", ""}};
            }
        } // namespace

        class Small_NpyPrefixRefusals : public ::testing::TestWithParam<PrefixRow>
        {
        };

        TEST_P(Small_NpyPrefixRefusals, ABadMagicVersionOrTruncationIsRefused)
        {
            const PrefixRow row = GetParam();

            const IOStatus status = refusal_of([&]()
            {
                static_cast<void>(parse_npy_prefix(row.bytes.data(), row.size, K_PATH));
            });

            EXPECT_EQ(status, row.status);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_NpyPrefixRefusals, ::testing::ValuesIn(refused_prefix_rows()),
                                 [](const ::testing::TestParamInfo<PrefixRow>& info)
                                 { return std::string(info.param.name); });

        TEST(Small_NpyPrefix, VersionOneKeepsItsLengthInTwoLittleEndianBytes)
        {
            const std::vector<unsigned char> bytes = npy_prefix(1, 0, 0x1234);
            const NpyPrefix prefix = parse_npy_prefix(bytes.data(), bytes.size(), K_PATH);

            EXPECT_EQ(prefix.header_bytes, std::size_t{0x1234});
            EXPECT_EQ(prefix.prefix_bytes, std::size_t{10});
        }

        TEST(Small_NpyPrefix, VersionsTwoAndThreeKeepTheirLengthInFourBytes)
        {
            for (const unsigned char major : {2, 3})
            {
                const std::vector<unsigned char> bytes = npy_prefix(major, 0, 0x012345);
                const NpyPrefix prefix = parse_npy_prefix(bytes.data(), bytes.size(), K_PATH);

                EXPECT_EQ(prefix.header_bytes, std::size_t{0x012345});
                EXPECT_EQ(prefix.prefix_bytes, std::size_t{12});
            }
        }

        TEST(Small_NpyPrefix, ALengthAboveTheCapIsMalformed)
        {
            const std::vector<unsigned char> bytes = npy_prefix(2, 0, K_MAX_NPY_HEADER_BYTES + 1);

            EXPECT_EQ(refusal_of([&]()
                      { static_cast<void>(parse_npy_prefix(bytes.data(), bytes.size(), K_PATH)); }),
                      IOStatus::MalformedFile);
        }

        TEST(Small_NpyPrefix, ALengthAtTheCapIsAccepted)
        {
            const std::vector<unsigned char> bytes = npy_prefix(2, 0, K_MAX_NPY_HEADER_BYTES);

            EXPECT_EQ(parse_npy_prefix(bytes.data(), bytes.size(), K_PATH).header_bytes,
                      K_MAX_NPY_HEADER_BYTES);
        }

        TEST(Small_NpyPrefix, BytesBeyondThePrefixAreIgnored)
        {
            std::vector<unsigned char> bytes = npy_prefix(1, 0, 64);
            bytes.push_back(0xFF);
            bytes.push_back(0xFF);

            EXPECT_EQ(parse_npy_prefix(bytes.data(), bytes.size(), K_PATH).prefix_bytes,
                      std::size_t{10});
        }

        TEST(Small_NpyHeader, ADictionaryGivesItsThreeFields)
        {
            const NpyHeader header = parse_npy_header(K_VALID_DICT, K_PATH);

            EXPECT_EQ(header.descr, "<f4");
            EXPECT_FALSE(header.fortran_order);
            EXPECT_EQ(header.shape, (std::vector<int64_t>{2, 3}));
        }

        TEST(Small_NpyHeader, NumpyPaddingAndATrailingNewlineAreSkipped)
        {
            const std::string padded = K_VALID_DICT + std::string(40, ' ') + "\n";

            EXPECT_EQ(parse_npy_header(padded, K_PATH).shape, (std::vector<int64_t>{2, 3}));
        }

        TEST(Small_NpyHeader, TheKeysMayComeInAnyOrderAndInDoubleQuotes)
        {
            const NpyHeader header = parse_npy_header(
                "{\"shape\": (4,), \"fortran_order\": True, \"descr\": \"<i8\"}", K_PATH);

            EXPECT_EQ(header.descr, "<i8");
            EXPECT_TRUE(header.fortran_order);
            EXPECT_EQ(header.shape, (std::vector<int64_t>{4}));
        }

        TEST(Small_NpyHeader, ARankZeroShapeIsAnEmptyTuple)
        {
            EXPECT_TRUE(parse_npy_header("{'descr': '<f4', 'fortran_order': False, 'shape': (), }", K_PATH)
                            .shape.empty());
        }

        class Small_NpyMalformedDictionaries : public ::testing::TestWithParam<DictRow>
        {
        };

        TEST_P(Small_NpyMalformedDictionaries, AreRefusedAsMalformed)
        {
            const DictRow row = GetParam();

            EXPECT_EQ(refusal_of([&]() { static_cast<void>(parse_npy_header(row.text, K_PATH)); }),
                      IOStatus::MalformedFile);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_NpyMalformedDictionaries,
                                 ::testing::ValuesIn(malformed_dict_rows()),
                                 [](const ::testing::TestParamInfo<DictRow>& info)
                                 { return std::string(info.param.name); });

        TEST(Small_NpyHeader, TheRefusalNamesTheFileAndTheByte)
        {
            try
            {
                static_cast<void>(parse_npy_header("{'descr': 5}", K_PATH));
                FAIL() << "the header was accepted";
            }
            catch (const HeaderRefusal& refusal)
            {
                EXPECT_NE(refusal.message.find(K_PATH), std::string::npos);
                EXPECT_NE(refusal.message.find("byte"), std::string::npos);
            }
        }

        TEST(Small_NpyMatch, AHeaderThatEqualsTheExpectationPasses)
        {
            EXPECT_EQ(refusal_of([&]()
                      { require_npy_header_matches(parse_npy_header(K_VALID_DICT, K_PATH),
                                                   expected_npy(), K_PATH); }),
                      IOStatus::Ok);
        }

        TEST(Small_NpyMatch, AnotherDtypeIsADtypeMismatch)
        {
            NpyExpectation expected = expected_npy();
            expected.descr = "<f8";

            EXPECT_EQ(refusal_of([&]()
                      { require_npy_header_matches(parse_npy_header(K_VALID_DICT, K_PATH), expected,
                                                   K_PATH); }),
                      IOStatus::DtypeMismatch);
        }

        TEST(Small_NpyMatch, AnotherShapeIsAShapeMismatch)
        {
            NpyExpectation expected = expected_npy();
            expected.shape = {3, 2};

            EXPECT_EQ(refusal_of([&]()
                      { require_npy_header_matches(parse_npy_header(K_VALID_DICT, K_PATH), expected,
                                                   K_PATH); }),
                      IOStatus::ShapeMismatch);
        }

        TEST(Small_NpyMatch, AnotherRankIsAShapeMismatch)
        {
            NpyExpectation expected = expected_npy();
            expected.shape = {2, 3, 1};

            EXPECT_EQ(refusal_of([&]()
                      { require_npy_header_matches(parse_npy_header(K_VALID_DICT, K_PATH), expected,
                                                   K_PATH); }),
                      IOStatus::ShapeMismatch);
        }

        TEST(Small_NpyMatch, AnotherOrderIsALayoutMismatchFromRankTwoUp)
        {
            NpyExpectation expected = expected_npy();
            expected.column_major = true;

            EXPECT_EQ(refusal_of([&]()
                      { require_npy_header_matches(parse_npy_header(K_VALID_DICT, K_PATH), expected,
                                                   K_PATH); }),
                      IOStatus::LayoutMismatch);
        }

        TEST(Small_NpyMatch, TheOrderOfARankOneArrayDoesNotMatter)
        {
            NpyExpectation expected;
            expected.descr = "<f4";
            expected.column_major = true;
            expected.shape = {6};
            const NpyHeader header =
                parse_npy_header("{'descr': '<f4', 'fortran_order': False, 'shape': (6,), }", K_PATH);

            EXPECT_EQ(refusal_of([&]() { require_npy_header_matches(header, expected, K_PATH); }),
                      IOStatus::Ok);
        }

        TEST(Small_NpyMatch, TheDtypeIsCheckedBeforeTheOrderAndTheOrderBeforeTheShape)
        {
            NpyExpectation expected;
            expected.descr = "<f8";
            expected.column_major = true;
            expected.shape = {9, 9};

            EXPECT_EQ(refusal_of([&]()
                      { require_npy_header_matches(parse_npy_header(K_VALID_DICT, K_PATH), expected,
                                                   K_PATH); }),
                      IOStatus::DtypeMismatch);

            expected.descr = "<f4";
            EXPECT_EQ(refusal_of([&]()
                      { require_npy_header_matches(parse_npy_header(K_VALID_DICT, K_PATH), expected,
                                                   K_PATH); }),
                      IOStatus::LayoutMismatch);
        }

        namespace
        {
            /** @brief Returns a header for a float32 2 x 3 row-major tensor of dtype id 5 and layout id 1. */
            SushiHeader valid_sushi_header()
            {
                SushiHeader header{};
                std::string("SUSHI").copy(header.magic, 5);
                header.version = K_SUSHI_FORMAT_VERSION;
                header.rank = 2;
                header.shape[0] = 2;
                header.shape[1] = 3;
                header.dtype = 5;
                header.layout = 1;
                return header;
            }

            /** @brief Returns the expectation valid_sushi_header() meets. */
            SushiExpectation expected_sushi()
            {
                SushiExpectation expected;
                expected.rank = 2;
                expected.shape = {2, 3};
                expected.dtype = 5;
                expected.dtype_name = "Float32";
                expected.layout = 1;
                return expected;
            }

            /** @brief Returns the status a .sushi header check refused with. */
            IOStatus sushi_status(const SushiHeader& header, const SushiExpectation& expected)
            {
                return refusal_of([&]() { require_sushi_header_matches(header, expected, K_PATH); });
            }
        } // namespace

        TEST(Small_SushiHeader, AHeaderThatEqualsTheExpectationPasses)
        {
            EXPECT_EQ(sushi_status(valid_sushi_header(), expected_sushi()), IOStatus::Ok);
        }

        TEST(Small_SushiHeader, AnotherMagicIsMalformed)
        {
            SushiHeader header = valid_sushi_header();
            header.magic[0] = 'X';

            EXPECT_EQ(sushi_status(header, expected_sushi()), IOStatus::MalformedFile);
        }

        TEST(Small_SushiHeader, AnotherFormatVersionIsMalformed)
        {
            SushiHeader header = valid_sushi_header();
            header.version = K_SUSHI_FORMAT_VERSION + 1;

            EXPECT_EQ(sushi_status(header, expected_sushi()), IOStatus::MalformedFile);
        }

        TEST(Small_SushiHeader, AnotherRankIsAShapeMismatch)
        {
            SushiHeader header = valid_sushi_header();
            header.rank = 3;

            EXPECT_EQ(sushi_status(header, expected_sushi()), IOStatus::ShapeMismatch);
        }

        TEST(Small_SushiHeader, AnotherDimensionIsAShapeMismatch)
        {
            SushiHeader header = valid_sushi_header();
            header.shape[1] = 4;

            EXPECT_EQ(sushi_status(header, expected_sushi()), IOStatus::ShapeMismatch);
        }

        TEST(Small_SushiHeader, AnotherDtypeIsADtypeMismatch)
        {
            SushiHeader header = valid_sushi_header();
            header.dtype = 6;

            EXPECT_EQ(sushi_status(header, expected_sushi()), IOStatus::DtypeMismatch);
        }

        TEST(Small_SushiHeader, AnotherLayoutIsALayoutMismatch)
        {
            SushiHeader header = valid_sushi_header();
            header.layout = 0;

            EXPECT_EQ(sushi_status(header, expected_sushi()), IOStatus::LayoutMismatch);
        }

        TEST(Small_SushiHeader, TheMagicIsCheckedBeforeTheVersionAndTheVersionBeforeTheShape)
        {
            SushiHeader header = valid_sushi_header();
            header.magic[0] = 'X';
            header.version = K_SUSHI_FORMAT_VERSION + 1;
            header.rank = 3;

            EXPECT_EQ(sushi_status(header, expected_sushi()), IOStatus::MalformedFile);

            header.magic[0] = 'S';
            EXPECT_EQ(sushi_status(header, expected_sushi()), IOStatus::MalformedFile);

            header.version = K_SUSHI_FORMAT_VERSION;
            EXPECT_EQ(sushi_status(header, expected_sushi()), IOStatus::ShapeMismatch);
        }
    } // namespace Test
} // namespace SushiBLAS
