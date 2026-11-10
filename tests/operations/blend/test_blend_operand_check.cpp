/**************************************************************************/
/* test_blend_operand_check.cpp                                           */
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
 * @file test_blend_operand_check.cpp
 * @brief Tests that check_blend_operands accepts a well-formed blend call and refuses each
 *        malformed one naming the operand, on operands built by hand.
 * @author Mustafa Garip
 */

#include <SushiBLAS/blend/blend_operand_check.hpp>

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
            constexpr const char* K_NAME = "blend.blend_transforms";

            /// @brief The made-up allocation of each operand of a well-formed call.
            constexpr std::size_t K_POINTS_ALLOCATION = 1;
            constexpr std::size_t K_TRANSFORMS_ALLOCATION = 2;
            constexpr std::size_t K_INDICES_ALLOCATION = 3;
            constexpr std::size_t K_WEIGHTS_ALLOCATION = 4;
            constexpr std::size_t K_OUT_ALLOCATION = 5;

            /** @brief Holds the five operands of a blend of two points by three transforms, two influences each. */
            struct BlendOperands
            {
                HandBuiltOperand points;
                HandBuiltOperand transforms;
                HandBuiltOperand indices;
                HandBuiltOperand weights;
                HandBuiltOperand out;
            };

            /** @brief Returns the operands of a well-formed call in the element dtype. */
            BlendOperands well_formed(Core::DataType element = Core::DataType::Float32)
            {
                return BlendOperands{
                    HandBuiltOperand({2, 3}, element, K_POINTS_ALLOCATION),
                    HandBuiltOperand({3, 3, 4}, element, K_TRANSFORMS_ALLOCATION),
                    HandBuiltOperand({2, 2}, Core::DataType::Int32, K_INDICES_ALLOCATION),
                    HandBuiltOperand({2, 2}, element, K_WEIGHTS_ALLOCATION),
                    HandBuiltOperand({2, 3}, element, K_OUT_ALLOCATION)};
            }

            /** @brief Runs the check on the operands. */
            void check(const BlendOperands& b)
            {
                Internal::check_blend_operands(b.points.placed(), b.transforms.placed(),
                                               b.indices.placed(), b.weights.placed(),
                                               b.out.placed(), K_NAME);
            }

            /** @brief Names an operand of a call and the member that holds it. */
            struct OperandRow
            {
                const char* name;                         /**< The name a message gives the operand. */
                HandBuiltOperand BlendOperands::* member; /**< Where the operand sits in the call. */
            };

            /** @brief Returns one row for each operand of a call. */
            std::vector<OperandRow> every_operand()
            {
                return {{"points", &BlendOperands::points},
                        {"transforms", &BlendOperands::transforms},
                        {"indices", &BlendOperands::indices},
                        {"weights", &BlendOperands::weights},
                        {"out", &BlendOperands::out}};
            }

            /** @brief Holds one spoiled call and the operand its refusal must name. */
            struct SpoilRow
            {
                const char* operand;                       /**< The operand the refusal names. */
                std::function<void(BlendOperands&)> spoil; /**< Turns the call into a malformed one. */
            };

            /** @brief Returns a spoil that replaces one operand with another of dims, dtype and allocation. */
            std::function<void(BlendOperands&)> replace(HandBuiltOperand BlendOperands::* member,
                                                        std::vector<int64_t> dims,
                                                        Core::DataType dtype,
                                                        std::size_t allocation)
            {
                return [=](BlendOperands& b)
                { b.*member = HandBuiltOperand(dims, dtype, allocation); };
            }

            /** @brief Expects every row to turn the well-formed call into one the check refuses naming the row's operand. */
            void expect_every_row_refused(const std::vector<SpoilRow>& rows)
            {
                for (const SpoilRow& row : rows)
                {
                    SCOPED_TRACE(row.operand);
                    BlendOperands b = well_formed();
                    row.spoil(b);
                    expect_refusal([&] { check(b); }, K_NAME, row.operand);
                }
            }
        } // namespace

        /** @brief Tests that a well-formed call passes and the same call with an overlapping output does not. */
        TEST(Small_BlendOperandCheck, AWellFormedCallIsAcceptedUntilItsOutputOverlapsAnInput)
        {
            BlendOperands overlapping = well_formed();
            overlapping.out.in_allocation(K_POINTS_ALLOCATION);

            EXPECT_NO_THROW(check(well_formed()));
            EXPECT_THROW(check(overlapping), std::invalid_argument);
        }

        /** @brief Tests that a position of three values and a homogeneous point of four pass, and two or five do not. */
        TEST(Small_BlendOperandCheck, ThreeOrFourValuesPerPointAreAcceptedAndTwoOrFiveAreRefused)
        {
            const Core::DataType real = Core::DataType::Float32;
            for (const int64_t values : {3, 4, 2, 5})
            {
                SCOPED_TRACE(values);
                BlendOperands b = well_formed();
                b.points = HandBuiltOperand({2, values}, real, K_POINTS_ALLOCATION);
                b.transforms = HandBuiltOperand({3, 4, 4}, real, K_TRANSFORMS_ALLOCATION);
                b.out = HandBuiltOperand({2, values}, real, K_OUT_ALLOCATION);

                if (values == 3 || values == 4)
                    EXPECT_NO_THROW(check(b));
                else
                    expect_refusal([&] { check(b); }, K_NAME, "points");
            }
        }

        /** @brief Tests that INT32, INT64 and UINT32 indices pass and a FLOAT32 index does not. */
        TEST(Small_BlendOperandCheck, EachIntegerIndexTypeIsAcceptedAndAFloatIndexIsRefused)
        {
            for (const Core::DataType index : {Core::DataType::Int32, Core::DataType::Int64,
                                               Core::DataType::UInt32, Core::DataType::Float32})
            {
                SCOPED_TRACE(Core::to_string(index));
                BlendOperands b = well_formed();
                b.indices = HandBuiltOperand({2, 2}, index, K_INDICES_ALLOCATION);

                if (index == Core::DataType::Float32)
                    expect_refusal([&] { check(b); }, K_NAME, "indices");
                else
                    EXPECT_NO_THROW(check(b));
            }
        }

        /** @brief Tests that a dtype the rule does not allow is refused naming the operand. */
        TEST(Small_BlendOperandCheck, ADtypeOutsideTheRuleIsRefusedNamingTheOperand)
        {
            const Core::DataType wide = Core::DataType::Float64;

            expect_every_row_refused(
                {{"transforms", replace(&BlendOperands::transforms, {3, 3, 4}, wide,
                                        K_TRANSFORMS_ALLOCATION)},
                 {"weights", replace(&BlendOperands::weights, {2, 2}, wide, K_WEIGHTS_ALLOCATION)},
                 {"out", replace(&BlendOperands::out, {2, 3}, wide, K_OUT_ALLOCATION)},
                 {"indices", replace(&BlendOperands::indices, {2, 2}, Core::DataType::Float32,
                                     K_INDICES_ALLOCATION)}});
        }

        /** @brief Tests that a rank, an extent or a count the rule does not allow is refused naming the operand. */
        TEST(Small_BlendOperandCheck, AShapeOutsideTheRuleIsRefusedNamingTheOperand)
        {
            const Core::DataType real = Core::DataType::Float32;
            const Core::DataType index = Core::DataType::Int32;

            expect_every_row_refused({
                {"points", replace(&BlendOperands::points, {6}, real, K_POINTS_ALLOCATION)},
                {"transforms", replace(&BlendOperands::transforms, {12}, real, K_TRANSFORMS_ALLOCATION)},
                {"indices", replace(&BlendOperands::indices, {4}, index, K_INDICES_ALLOCATION)},
                {"weights", replace(&BlendOperands::weights, {4}, real, K_WEIGHTS_ALLOCATION)},
                {"out", replace(&BlendOperands::out, {6}, real, K_OUT_ALLOCATION)},
                {"points", replace(&BlendOperands::points, {2, 2}, real, K_POINTS_ALLOCATION)},
                {"points", replace(&BlendOperands::points, {2, 5}, real, K_POINTS_ALLOCATION)},
                {"transforms", replace(&BlendOperands::transforms, {3, 2, 4}, real, K_TRANSFORMS_ALLOCATION)},
                {"transforms", replace(&BlendOperands::transforms, {3, 3, 3}, real, K_TRANSFORMS_ALLOCATION)},
                {"transforms", replace(&BlendOperands::transforms, {0, 3, 4}, real, K_TRANSFORMS_ALLOCATION)},
                {"indices", replace(&BlendOperands::indices, {3, 2}, index, K_INDICES_ALLOCATION)},
                {"weights", replace(&BlendOperands::weights, {2, 3}, real, K_WEIGHTS_ALLOCATION)},
                {"out", replace(&BlendOperands::out, {2, 4}, real, K_OUT_ALLOCATION)},
                {"indices", [=](BlendOperands& b)
                 {
                     b.indices = HandBuiltOperand({2, 0}, index, K_INDICES_ALLOCATION);
                     b.weights = HandBuiltOperand({2, 0}, real, K_WEIGHTS_ALLOCATION);
                 }},
                {"points", [=](BlendOperands& b)
                 {
                     b.points = HandBuiltOperand({0, 3}, real, K_POINTS_ALLOCATION);
                     b.indices = HandBuiltOperand({0, 2}, index, K_INDICES_ALLOCATION);
                     b.weights = HandBuiltOperand({0, 2}, real, K_WEIGHTS_ALLOCATION);
                     b.out = HandBuiltOperand({0, 3}, real, K_OUT_ALLOCATION);
                 }}});
        }

        /** @brief Tests that an operand with elements and no storage is refused naming it. */
        TEST(Small_BlendOperandCheck, AnOperandWithoutStorageIsRefusedNamingIt)
        {
            for (const OperandRow& row : every_operand())
            {
                SCOPED_TRACE(row.name);
                BlendOperands b = well_formed();
                (b.*row.member).without_storage();

                expect_refusal([&] { check(b); }, K_NAME, row.name);
            }
        }

        /** @brief Tests that a strided view and a column-major tensor are refused naming the operand. */
        TEST(Small_BlendOperandCheck, AViewAndAColumnMajorOperandAreRefusedNamingIt)
        {
            for (const OperandRow& row : every_operand())
            {
                SCOPED_TRACE(row.name);
                BlendOperands strided = well_formed();
                // Stride 2 is the dense stride of the {2, 2} operands; stride 1 is no operand's.
                (strided.*row.member).with_stride(0, 1);
                BlendOperands column_major = well_formed();
                (column_major.*row.member).column_major();

                expect_refusal([&] { check(strided); }, K_NAME, row.name);
                expect_refusal([&] { check(column_major); }, K_NAME, row.name);
            }
        }

        /** @brief Tests that a device without the aspect of the dtype is refused naming the operand. */
        TEST(Small_BlendOperandCheck, ADeviceWithoutTheAspectOfTheDtypeIsRefusedNamingTheOperand)
        {
            for (const OperandRow& row : every_operand())
            {
                if (row.member == &BlendOperands::indices)
                    continue;

                SCOPED_TRACE(row.name);
                BlendOperands wide = well_formed(Core::DataType::Float64);
                (wide.*row.member).on_device_without_fp64();
                BlendOperands narrow = well_formed(Core::DataType::Half);
                (narrow.*row.member).on_device_without_fp16();

                EXPECT_THROW(check(wide), UnsupportedExecution);
                EXPECT_THROW(check(narrow), UnsupportedExecution);
                expect_refusal([&] { check(wide); }, K_NAME, row.name);
                expect_refusal([&] { check(narrow); }, K_NAME, row.name);
            }
        }

        /** @brief Tests that an output that shares a byte with an input is refused naming the output. */
        TEST(Small_BlendOperandCheck, AnOutputThatSharesBytesWithAnInputIsRefusedNamingTheOutput)
        {
            for (const OperandRow& row : every_operand())
            {
                if (row.member == &BlendOperands::out)
                    continue;

                SCOPED_TRACE(row.name);
                BlendOperands b = well_formed();
                b.out.in_allocation((b.*row.member).allocation());

                expect_refusal([&] { check(b); }, K_NAME, "out");
            }
        }

        /** @brief Tests that an output starting where the points end passes and one a byte earlier does not. */
        TEST(Small_BlendOperandCheck, AnOutputStartingWhereAnInputEndsIsAcceptedAndOneByteEarlierIsRefused)
        {
            const std::size_t points_bytes = 6 * sizeof(float);
            BlendOperands adjacent = well_formed();
            adjacent.out.in_allocation(K_POINTS_ALLOCATION).at_byte_offset(points_bytes);
            BlendOperands overlapping = well_formed();
            overlapping.out.in_allocation(K_POINTS_ALLOCATION).at_byte_offset(points_bytes - 1);

            EXPECT_NO_THROW(check(adjacent));
            expect_refusal([&] { check(overlapping); }, K_NAME, "out");
        }
    } // namespace Test
} // namespace SushiBLAS
