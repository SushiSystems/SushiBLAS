/**************************************************************************/
/* test_convolve_operand_check.cpp                                        */
/**************************************************************************/
/*                          This file is part of:                         */
/*                               SushiSkills                              */
/*               https://github.com/SushiSystems/SushiSkills              */
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
 * @file test_convolve_operand_check.cpp
 * @brief Tests that check_convolve_operands accepts a well-formed convolution call and refuses
 *        each malformed one naming the operand, on operands built by hand.
 * @author Mustafa Garip
 */

#include <SushiBLAS/signal/convolve_operand_check.hpp>

#include <vector>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <gtest/gtest.h>
#include "support/forced_mode_checks.hpp"
#include "support/hand_built_operand.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief The operation id every message names.
            constexpr const char* K_NAME = "signal.convolve_1d";

            /// @brief The made-up allocation of each operand of a well-formed call.
            constexpr std::size_t K_SIGNAL_ALLOCATION = 1;
            constexpr std::size_t K_TAPS_ALLOCATION = 2;
            constexpr std::size_t K_OUT_ALLOCATION = 3;

            /** @brief Holds the three operands of a convolution of two rows of eight samples by three taps. */
            struct ConvolveOperands
            {
                HandBuiltOperand signal;
                HandBuiltOperand taps;
                HandBuiltOperand out;
            };

            /** @brief Returns the operands of a well-formed call in the element dtype. */
            ConvolveOperands well_formed(Core::DataType element = Core::DataType::Float32)
            {
                return ConvolveOperands{HandBuiltOperand({2, 8}, element, K_SIGNAL_ALLOCATION),
                                        HandBuiltOperand({3}, element, K_TAPS_ALLOCATION),
                                        HandBuiltOperand({2, 6}, element, K_OUT_ALLOCATION)};
            }

            /** @brief Runs the check on the operands and returns the number of rows it reports. */
            int64_t check(const ConvolveOperands& c)
            {
                return Internal::check_convolve_operands(c.signal.placed(), c.taps.placed(),
                                                         c.out.placed(), K_NAME);
            }

            /** @brief Names an operand of a call and the member that holds it. */
            struct OperandRow
            {
                const char* name;                           /**< The name a message gives the operand. */
                HandBuiltOperand ConvolveOperands::* member; /**< Where the operand sits in the call. */
            };

            /** @brief Returns one row for each operand of a call. */
            std::vector<OperandRow> every_operand()
            {
                return {{"signal", &ConvolveOperands::signal},
                        {"taps", &ConvolveOperands::taps},
                        {"out", &ConvolveOperands::out}};
            }

            /** @brief Holds one spoiled call and the operand its refusal must name. */
            struct SpoilRow
            {
                const char* operand;                          /**< The operand the refusal names. */
                std::function<void(ConvolveOperands&)> spoil; /**< Turns the call into a malformed one. */
            };

            /** @brief Returns a spoil that replaces one operand with another of dims, dtype and allocation. */
            std::function<void(ConvolveOperands&)> replace(HandBuiltOperand ConvolveOperands::* member,
                                                           std::vector<int64_t> dims,
                                                           Core::DataType dtype,
                                                           std::size_t allocation)
            {
                return [=](ConvolveOperands& c)
                { c.*member = HandBuiltOperand(dims, dtype, allocation); };
            }

            /** @brief Expects every row to turn the well-formed call into one the check refuses naming the row's operand. */
            void expect_every_row_refused(const std::vector<SpoilRow>& rows)
            {
                for (const SpoilRow& row : rows)
                {
                    SCOPED_TRACE(row.operand);
                    ConvolveOperands c = well_formed();
                    row.spoil(c);
                    expect_refusal([&] { check(c); }, K_NAME, row.operand);
                }
            }
        } // namespace

        /** @brief Tests that a well-formed call passes and the same call with an overlapping output does not. */
        TEST(Small_ConvolveOperandCheck, AWellFormedCallIsAcceptedUntilItsOutputOverlapsTheSignal)
        {
            ConvolveOperands overlapping = well_formed();
            overlapping.out.in_allocation(K_SIGNAL_ALLOCATION);

            EXPECT_NO_THROW(check(well_formed()));
            EXPECT_THROW(check(overlapping), std::invalid_argument);
        }

        /** @brief Tests that the check reports one row per leading index of the signal. */
        TEST(Small_ConvolveOperandCheck, TheRowCountIsTheProductOfTheLeadingDimensions)
        {
            ConvolveOperands one_row = well_formed();
            one_row.signal = HandBuiltOperand({8}, Core::DataType::Float32, K_SIGNAL_ALLOCATION);
            one_row.out = HandBuiltOperand({6}, Core::DataType::Float32, K_OUT_ALLOCATION);
            ConvolveOperands six_rows = well_formed();
            six_rows.signal = HandBuiltOperand({2, 3, 9}, Core::DataType::Float32, K_SIGNAL_ALLOCATION);
            six_rows.taps = HandBuiltOperand({2}, Core::DataType::Float32, K_TAPS_ALLOCATION);
            six_rows.out = HandBuiltOperand({2, 3, 8}, Core::DataType::Float32, K_OUT_ALLOCATION);

            EXPECT_EQ(check(one_row), 1);
            EXPECT_EQ(check(well_formed()), 2);
            EXPECT_EQ(check(six_rows), 6);
        }

        /** @brief Tests that taps as long as a row pass, and one tap more is refused naming the signal. */
        TEST(Small_ConvolveOperandCheck, TapsAsLongAsTheRowAreAcceptedAndOneTapMoreIsRefused)
        {
            ConvolveOperands as_long = well_formed();
            as_long.taps = HandBuiltOperand({8}, Core::DataType::Float32, K_TAPS_ALLOCATION);
            as_long.out = HandBuiltOperand({2, 1}, Core::DataType::Float32, K_OUT_ALLOCATION);
            ConvolveOperands longer = well_formed();
            longer.taps = HandBuiltOperand({9}, Core::DataType::Float32, K_TAPS_ALLOCATION);
            longer.out = HandBuiltOperand({2, 0}, Core::DataType::Float32, K_OUT_ALLOCATION);

            EXPECT_NO_THROW(check(as_long));
            expect_refusal([&] { check(longer); }, K_NAME, "signal");
        }

        /** @brief Tests that a dtype the rule does not allow is refused naming the operand. */
        TEST(Small_ConvolveOperandCheck, ADtypeOutsideTheRuleIsRefusedNamingTheOperand)
        {
            const Core::DataType wide = Core::DataType::Float64;

            expect_every_row_refused(
                {{"taps", replace(&ConvolveOperands::taps, {3}, wide, K_TAPS_ALLOCATION)},
                 {"out", replace(&ConvolveOperands::out, {2, 6}, wide, K_OUT_ALLOCATION)}});
        }

        /** @brief Tests that a rank, an extent or a count the rule does not allow is refused naming the operand. */
        TEST(Small_ConvolveOperandCheck, AShapeOutsideTheRuleIsRefusedNamingTheOperand)
        {
            const Core::DataType real = Core::DataType::Float32;

            expect_every_row_refused({
                {"taps", replace(&ConvolveOperands::taps, {1, 3}, real, K_TAPS_ALLOCATION)},
                {"signal", replace(&ConvolveOperands::signal, {}, real, K_SIGNAL_ALLOCATION)},
                {"out", replace(&ConvolveOperands::out, {12}, real, K_OUT_ALLOCATION)},
                {"out", replace(&ConvolveOperands::out, {3, 6}, real, K_OUT_ALLOCATION)},
                {"out", replace(&ConvolveOperands::out, {2, 7}, real, K_OUT_ALLOCATION)},
                {"taps", [=](ConvolveOperands& c)
                 {
                     c.taps = HandBuiltOperand({0}, real, K_TAPS_ALLOCATION);
                     c.out = HandBuiltOperand({2, 9}, real, K_OUT_ALLOCATION);
                 }},
                {"signal", [=](ConvolveOperands& c)
                 {
                     c.taps = HandBuiltOperand({9}, real, K_TAPS_ALLOCATION);
                     c.out = HandBuiltOperand({2, 0}, real, K_OUT_ALLOCATION);
                 }}});
        }

        /** @brief Tests that an operand with elements and no storage is refused naming it. */
        TEST(Small_ConvolveOperandCheck, AnOperandWithoutStorageIsRefusedNamingIt)
        {
            for (const OperandRow& row : every_operand())
            {
                SCOPED_TRACE(row.name);
                ConvolveOperands c = well_formed();
                (c.*row.member).without_storage();

                expect_refusal([&] { check(c); }, K_NAME, row.name);
            }
        }

        /** @brief Tests that a strided view is refused naming the operand. */
        TEST(Small_ConvolveOperandCheck, AStridedViewIsRefusedNamingIt)
        {
            for (const OperandRow& row : every_operand())
            {
                SCOPED_TRACE(row.name);
                ConvolveOperands c = well_formed();
                (c.*row.member).with_stride(0, 2);

                expect_refusal([&] { check(c); }, K_NAME, row.name);
            }
        }

        /** @brief Tests that a column-major signal or output is refused, and that column-major taps of rank one are not. */
        TEST(Small_ConvolveOperandCheck, ARankTwoOperandInColumnMajorIsRefusedAndRankOneTapsAreNot)
        {
            ConvolveOperands signal = well_formed();
            signal.signal.column_major();
            ConvolveOperands out = well_formed();
            out.out.column_major();
            ConvolveOperands taps = well_formed();
            taps.taps.column_major();

            expect_refusal([&] { check(signal); }, K_NAME, "signal");
            expect_refusal([&] { check(out); }, K_NAME, "out");
            EXPECT_NO_THROW(check(taps));
        }

        /** @brief Tests that a device without the aspect of the dtype is refused naming the operand. */
        TEST(Small_ConvolveOperandCheck, ADeviceWithoutTheAspectOfTheDtypeIsRefusedNamingTheOperand)
        {
            for (const OperandRow& row : every_operand())
            {
                SCOPED_TRACE(row.name);
                ConvolveOperands wide = well_formed(Core::DataType::Float64);
                (wide.*row.member).on_device_without_fp64();
                ConvolveOperands narrow = well_formed(Core::DataType::Half);
                (narrow.*row.member).on_device_without_fp16();

                EXPECT_THROW(check(wide), UnsupportedExecution);
                EXPECT_THROW(check(narrow), UnsupportedExecution);
                expect_refusal([&] { check(wide); }, K_NAME, row.name);
                expect_refusal([&] { check(narrow); }, K_NAME, row.name);
            }
        }

        /** @brief Tests that an output that shares a byte with an input is refused naming the output. */
        TEST(Small_ConvolveOperandCheck, AnOutputThatSharesBytesWithAnInputIsRefusedNamingTheOutput)
        {
            for (const OperandRow& row : every_operand())
            {
                if (row.member == &ConvolveOperands::out)
                    continue;

                SCOPED_TRACE(row.name);
                ConvolveOperands c = well_formed();
                c.out.in_allocation((c.*row.member).allocation());

                expect_refusal([&] { check(c); }, K_NAME, "out");
            }
        }

        /** @brief Tests that an output starting where the signal ends passes and one a byte earlier does not. */
        TEST(Small_ConvolveOperandCheck, AnOutputStartingWhereTheSignalEndsIsAcceptedAndOneByteEarlierIsRefused)
        {
            const std::size_t signal_bytes = 16 * sizeof(float);
            ConvolveOperands adjacent = well_formed();
            adjacent.out.in_allocation(K_SIGNAL_ALLOCATION).at_byte_offset(signal_bytes);
            ConvolveOperands overlapping = well_formed();
            overlapping.out.in_allocation(K_SIGNAL_ALLOCATION).at_byte_offset(signal_bytes - 1);

            EXPECT_NO_THROW(check(adjacent));
            expect_refusal([&] { check(overlapping); }, K_NAME, "out");
        }
    } // namespace Test
} // namespace SushiBLAS
