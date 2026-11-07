/**************************************************************************/
/* test_linalg_operand_check.cpp                                          */
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
 * @file test_linalg_operand_check.cpp
 * @brief Tests that the operand checks of lu_factor, lu_solve and gmres accept a well-formed
 *        call and refuse each malformed one naming the operand, on operands built by hand.
 * @author Mustafa Garip
 */

#include <SushiBLAS/experimental/lapack/linalg_operand_check.hpp>

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
            /// @brief The made-up allocation of each operand, in the order the operands are declared.
            constexpr std::size_t K_FIRST = 1;
            constexpr std::size_t K_SECOND = 2;
            constexpr std::size_t K_THIRD = 3;
            constexpr std::size_t K_FOURTH = 4;
            constexpr std::size_t K_FIFTH = 5;

            /// @brief The operation ids the messages name.
            constexpr const char* K_FACTOR = "linalg.lu_factor";
            constexpr const char* K_SOLVE = "linalg.lu_solve";
            constexpr const char* K_GMRES = "linalg.gmres";

            /** @brief Holds the operands of a factorisation of one matrix of order two. */
            struct FactorCall
            {
                HandBuiltOperand matrices;
                HandBuiltOperand pivots;
                HandBuiltOperand singular;
                std::size_t n = 2;
                std::size_t batch = 1;

                /** @brief Returns a well-formed call whose real operands hold element. */
                static FactorCall well_formed(Core::DataType element = Core::DataType::Float32)
                {
                    return FactorCall{HandBuiltOperand({4}, element, K_FIRST),
                                      HandBuiltOperand({2}, Core::DataType::UInt32, K_SECOND),
                                      HandBuiltOperand({1}, Core::DataType::UInt32, K_THIRD)};
                }

                /** @brief Runs the check of the operation name. */
                void check(const char* name) const
                {
                    Internal::check_lu_factor_operands(matrices.placed(), pivots.placed(),
                                                       singular.placed(), n, batch, name);
                }
            };

            /** @brief Holds the operands of a solve of one system of order two for one right-hand side. */
            struct SolveCall
            {
                HandBuiltOperand factors;
                HandBuiltOperand pivots;
                HandBuiltOperand rhs;
                std::size_t n = 2;
                std::size_t rhs_count = 1;
                std::size_t batch = 1;

                /** @brief Returns a well-formed call whose real operands hold element. */
                static SolveCall well_formed(Core::DataType element = Core::DataType::Float32)
                {
                    return SolveCall{HandBuiltOperand({4}, element, K_FIRST),
                                     HandBuiltOperand({2}, Core::DataType::UInt32, K_SECOND),
                                     HandBuiltOperand({2}, element, K_THIRD)};
                }

                /** @brief Runs the check of the operation name. */
                void check(const char* name) const
                {
                    Internal::check_lu_solve_operands(factors.placed(), pivots.placed(),
                                                      rhs.placed(), n, rhs_count, batch, name);
                }
            };

            /** @brief Holds the operands and the options of a GMRES solve of order two. */
            struct GmresCall
            {
                HandBuiltOperand matrix;
                HandBuiltOperand rhs;
                HandBuiltOperand solution;
                HandBuiltOperand iterations;
                HandBuiltOperand residual;
                GmresOptions options;

                /** @brief Returns a well-formed call whose real operands hold element. */
                static GmresCall well_formed(Core::DataType element = Core::DataType::Float32)
                {
                    return GmresCall{HandBuiltOperand({4}, element, K_FIRST),
                                     HandBuiltOperand({2}, element, K_SECOND),
                                     HandBuiltOperand({2}, element, K_THIRD),
                                     HandBuiltOperand({1}, Core::DataType::UInt32, K_FOURTH),
                                     HandBuiltOperand({1}, element, K_FIFTH),
                                     GmresOptions{}};
                }

                /** @brief Runs the check of the operation name and returns the order it reports. */
                std::size_t check(const char* name) const
                {
                    return Internal::check_gmres_operands(matrix.placed(), rhs.placed(),
                                                          solution.placed(), iterations.placed(),
                                                          residual.placed(), options, name);
                }
            };

            /** @brief Names an operand of a call and the member that holds it. */
            template<typename Call>
            struct OperandRow
            {
                const char* name;               /**< The name a message gives the operand. */
                HandBuiltOperand Call::* member; /**< Where the operand sits in the call. */
            };

            /** @brief Holds one spoiled call and the operand its refusal must name. */
            template<typename Call>
            struct SpoilRow
            {
                const char* operand;                 /**< The operand the refusal names. */
                std::function<void(Call&)> spoil;    /**< Turns the call into a malformed one. */
            };

            /** @brief Returns a spoil that replaces one operand with another of dims, dtype and allocation. */
            template<typename Call>
            std::function<void(Call&)> replace(HandBuiltOperand Call::* member,
                                               std::vector<int64_t> dims, Core::DataType dtype,
                                               std::size_t allocation)
            {
                return [=](Call& c) { c.*member = HandBuiltOperand(dims, dtype, allocation); };
            }

            /** @brief Expects every row to turn the well-formed call into one the check refuses naming the row's operand. */
            template<typename Call>
            void expect_every_row_refused(const char* op, const std::vector<SpoilRow<Call>>& rows)
            {
                for (const SpoilRow<Call>& row : rows)
                {
                    SCOPED_TRACE(row.operand);
                    Call c = Call::well_formed();
                    row.spoil(c);
                    expect_refusal([&] { c.check(op); }, op, row.operand);
                }
            }

            /** @brief Expects a call to be refused for every operand taken without storage. */
            template<typename Call>
            void expect_missing_storage_refused(const char* op,
                                                const std::vector<OperandRow<Call>>& operands)
            {
                for (const OperandRow<Call>& row : operands)
                {
                    SCOPED_TRACE(row.name);
                    Call c = Call::well_formed();
                    (c.*row.member).without_storage();
                    expect_refusal([&] { c.check(op); }, op, row.name);
                }
            }

            /** @brief Expects a call to be refused for every operand made a strided view. */
            template<typename Call>
            void expect_strided_view_refused(const char* op,
                                             const std::vector<OperandRow<Call>>& operands)
            {
                for (const OperandRow<Call>& row : operands)
                {
                    SCOPED_TRACE(row.name);
                    Call c = Call::well_formed();
                    (c.*row.member).with_stride(0, 2);
                    expect_refusal([&] { c.check(op); }, op, row.name);
                }
            }

            /** @brief Expects a call of FLOAT64 operands to be refused for each operand on a device without fp64. */
            template<typename Call>
            void expect_wide_operand_on_narrow_device_refused(
                const char* op, const std::vector<OperandRow<Call>>& operands)
            {
                for (const OperandRow<Call>& row : operands)
                {
                    SCOPED_TRACE(row.name);
                    Call c = Call::well_formed(Core::DataType::Float64);
                    (c.*row.member).on_device_without_fp64();

                    EXPECT_THROW(c.check(op), UnsupportedExecution);
                    expect_refusal([&] { c.check(op); }, op, row.name);
                }
            }

            /** @brief Expects the written operand to be refused, named, once it shares the allocation of another. */
            template<typename Call>
            void expect_overlap_refused(const char* op, const char* written_name,
                                        HandBuiltOperand Call::* written,
                                        HandBuiltOperand Call::* other)
            {
                Call c = Call::well_formed();
                (c.*written).in_allocation((c.*other).allocation());

                expect_refusal([&] { c.check(op); }, op, written_name);
            }
        } // namespace

        /** @brief Tests that a well-formed factorisation passes, in FLOAT32 and FLOAT64, until its pivots overlap the matrices. */
        TEST(Small_LuFactorOperandCheck, AWellFormedCallIsAcceptedUntilItsPivotsOverlapTheMatrices)
        {
            FactorCall overlapping = FactorCall::well_formed();
            overlapping.pivots.in_allocation(K_FIRST);

            EXPECT_NO_THROW(FactorCall::well_formed().check(K_FACTOR));
            EXPECT_NO_THROW(FactorCall::well_formed(Core::DataType::Float64).check(K_FACTOR));
            EXPECT_THROW(overlapping.check(K_FACTOR), std::invalid_argument);
        }

        /** @brief Tests that an order of zero is refused naming n. */
        TEST(Small_LuFactorOperandCheck, AnOrderOfZeroIsRefusedNamingN)
        {
            expect_every_row_refused<FactorCall>(
                K_FACTOR, {{"n", [](FactorCall& c) { c.n = 0; }}});
        }

        /** @brief Tests that a dtype the rule does not allow is refused naming the operand. */
        TEST(Small_LuFactorOperandCheck, ADtypeOutsideTheRuleIsRefusedNamingTheOperand)
        {
            expect_every_row_refused<FactorCall>(
                K_FACTOR,
                {{"matrices", replace<FactorCall>(&FactorCall::matrices, {4}, Core::DataType::Int32, K_FIRST)},
                 {"matrices", replace<FactorCall>(&FactorCall::matrices, {4}, Core::DataType::Half, K_FIRST)},
                 {"pivots", replace<FactorCall>(&FactorCall::pivots, {2}, Core::DataType::Float32, K_SECOND)},
                 {"singular", replace<FactorCall>(&FactorCall::singular, {1}, Core::DataType::Float32, K_THIRD)}});
        }

        /** @brief Tests that an operand of the wrong element count is refused naming it. */
        TEST(Small_LuFactorOperandCheck, AWrongElementCountIsRefusedNamingTheOperand)
        {
            expect_every_row_refused<FactorCall>(
                K_FACTOR,
                {{"matrices", replace<FactorCall>(&FactorCall::matrices, {5}, Core::DataType::Float32, K_FIRST)},
                 {"pivots", replace<FactorCall>(&FactorCall::pivots, {3}, Core::DataType::UInt32, K_SECOND)},
                 {"singular", replace<FactorCall>(&FactorCall::singular, {2}, Core::DataType::UInt32, K_THIRD)}});
        }

        /** @brief Tests that an operand without storage is refused naming it. */
        TEST(Small_LuFactorOperandCheck, AnOperandWithoutStorageIsRefusedNamingIt)
        {
            expect_missing_storage_refused<FactorCall>(
                K_FACTOR, {{"matrices", &FactorCall::matrices},
                           {"pivots", &FactorCall::pivots},
                           {"singular", &FactorCall::singular}});
        }

        /** @brief Tests that a strided view is refused naming the operand. */
        TEST(Small_LuFactorOperandCheck, AStridedViewIsRefusedNamingIt)
        {
            expect_strided_view_refused<FactorCall>(
                K_FACTOR, {{"matrices", &FactorCall::matrices}, {"pivots", &FactorCall::pivots}});
        }

        /** @brief Tests that a matrix on a device without fp64 is refused naming it. */
        TEST(Small_LuFactorOperandCheck, AMatrixOnADeviceWithoutFp64IsRefusedNamingIt)
        {
            expect_wide_operand_on_narrow_device_refused<FactorCall>(
                K_FACTOR, {{"matrices", &FactorCall::matrices}});
        }

        /** @brief Tests that pivots or flags that share a byte with another operand are refused naming them. */
        TEST(Small_LuFactorOperandCheck, AnOutputThatSharesBytesWithAnotherOperandIsRefusedNamingIt)
        {
            expect_overlap_refused<FactorCall>(K_FACTOR, "pivots", &FactorCall::pivots,
                                               &FactorCall::matrices);
            expect_overlap_refused<FactorCall>(K_FACTOR, "singular", &FactorCall::singular,
                                               &FactorCall::matrices);
            expect_overlap_refused<FactorCall>(K_FACTOR, "singular", &FactorCall::singular,
                                               &FactorCall::pivots);
        }

        /** @brief Tests that a well-formed solve passes, in FLOAT32 and FLOAT64, until its right-hand sides overlap the factors. */
        TEST(Small_LuSolveOperandCheck, AWellFormedCallIsAcceptedUntilItsRightHandSidesOverlapTheFactors)
        {
            SolveCall overlapping = SolveCall::well_formed();
            overlapping.rhs.in_allocation(K_FIRST);

            EXPECT_NO_THROW(SolveCall::well_formed().check(K_SOLVE));
            EXPECT_NO_THROW(SolveCall::well_formed(Core::DataType::Float64).check(K_SOLVE));
            EXPECT_THROW(overlapping.check(K_SOLVE), std::invalid_argument);
        }

        /** @brief Tests that an order of zero is refused naming n. */
        TEST(Small_LuSolveOperandCheck, AnOrderOfZeroIsRefusedNamingN)
        {
            expect_every_row_refused<SolveCall>(K_SOLVE, {{"n", [](SolveCall& c) { c.n = 0; }}});
        }

        /** @brief Tests that a dtype the rule does not allow is refused naming the operand. */
        TEST(Small_LuSolveOperandCheck, ADtypeOutsideTheRuleIsRefusedNamingTheOperand)
        {
            expect_every_row_refused<SolveCall>(
                K_SOLVE,
                {{"factors", replace<SolveCall>(&SolveCall::factors, {4}, Core::DataType::Int32, K_FIRST)},
                 {"rhs", replace<SolveCall>(&SolveCall::rhs, {2}, Core::DataType::Int32, K_THIRD)},
                 {"rhs", replace<SolveCall>(&SolveCall::rhs, {2}, Core::DataType::Float64, K_THIRD)},
                 {"pivots", replace<SolveCall>(&SolveCall::pivots, {2}, Core::DataType::Float32, K_SECOND)}});
        }

        /** @brief Tests that an operand of the wrong element count is refused naming it. */
        TEST(Small_LuSolveOperandCheck, AWrongElementCountIsRefusedNamingTheOperand)
        {
            expect_every_row_refused<SolveCall>(
                K_SOLVE,
                {{"factors", replace<SolveCall>(&SolveCall::factors, {5}, Core::DataType::Float32, K_FIRST)},
                 {"pivots", replace<SolveCall>(&SolveCall::pivots, {3}, Core::DataType::UInt32, K_SECOND)},
                 {"rhs", replace<SolveCall>(&SolveCall::rhs, {3}, Core::DataType::Float32, K_THIRD)}});
        }

        /** @brief Tests that an operand without storage is refused naming it. */
        TEST(Small_LuSolveOperandCheck, AnOperandWithoutStorageIsRefusedNamingIt)
        {
            expect_missing_storage_refused<SolveCall>(
                K_SOLVE, {{"factors", &SolveCall::factors},
                          {"pivots", &SolveCall::pivots},
                          {"rhs", &SolveCall::rhs}});
        }

        /** @brief Tests that a strided view is refused naming the operand. */
        TEST(Small_LuSolveOperandCheck, AStridedViewIsRefusedNamingIt)
        {
            expect_strided_view_refused<SolveCall>(
                K_SOLVE, {{"factors", &SolveCall::factors},
                          {"pivots", &SolveCall::pivots},
                          {"rhs", &SolveCall::rhs}});
        }

        /** @brief Tests that the factors or the right-hand sides on a device without fp64 are refused naming them. */
        TEST(Small_LuSolveOperandCheck, AnOperandOnADeviceWithoutFp64IsRefusedNamingIt)
        {
            expect_wide_operand_on_narrow_device_refused<SolveCall>(
                K_SOLVE, {{"factors", &SolveCall::factors}, {"rhs", &SolveCall::rhs}});
        }

        /** @brief Tests that right-hand sides that share a byte with the factors or the pivots are refused naming them. */
        TEST(Small_LuSolveOperandCheck, RightHandSidesThatShareBytesWithAnInputAreRefusedNamingThem)
        {
            expect_overlap_refused<SolveCall>(K_SOLVE, "rhs", &SolveCall::rhs, &SolveCall::factors);
            expect_overlap_refused<SolveCall>(K_SOLVE, "rhs", &SolveCall::rhs, &SolveCall::pivots);
        }

        /** @brief Tests that a well-formed solve passes and reports the order, in FLOAT32 and FLOAT64. */
        TEST(Small_GmresOperandCheck, AWellFormedCallIsAcceptedAndReportsTheOrder)
        {
            EXPECT_EQ(GmresCall::well_formed().check(K_GMRES), std::size_t{2});
            EXPECT_EQ(GmresCall::well_formed(Core::DataType::Float64).check(K_GMRES),
                      std::size_t{2});
        }

        /** @brief Tests that a block that divides the order passes and one that does not is refused. */
        TEST(Small_GmresOperandCheck, ABlockThatDividesTheOrderIsAcceptedAndOneThatDoesNotIsRefused)
        {
            GmresCall divides = GmresCall::well_formed();
            divides.options.block_size = 2;
            GmresCall does_not_divide = GmresCall::well_formed();
            does_not_divide.options.block_size = 3;

            EXPECT_NO_THROW(divides.check(K_GMRES));
            expect_refusal([&] { does_not_divide.check(K_GMRES); }, K_GMRES, "options");
        }

        /** @brief Tests that an option outside its rule is refused naming the options. */
        TEST(Small_GmresOperandCheck, AnOptionOutsideTheRuleIsRefusedNamingTheOptions)
        {
            expect_every_row_refused<GmresCall>(
                K_GMRES, {{"options", [](GmresCall& c) { c.options.restart = 0; }},
                          {"options", [](GmresCall& c) { c.options.max_iterations = 0; }},
                          {"options", [](GmresCall& c) { c.options.block_size = 0; }},
                          {"options", [](GmresCall& c) { c.options.block_size = 3; }}});
        }

        /** @brief Tests that a dtype the rule does not allow is refused naming the operand. */
        TEST(Small_GmresOperandCheck, ADtypeOutsideTheRuleIsRefusedNamingTheOperand)
        {
            expect_every_row_refused<GmresCall>(
                K_GMRES,
                {{"matrix", replace<GmresCall>(&GmresCall::matrix, {4}, Core::DataType::Int32, K_FIRST)},
                 {"matrix", replace<GmresCall>(&GmresCall::matrix, {4}, Core::DataType::Half, K_FIRST)},
                 {"rhs", replace<GmresCall>(&GmresCall::rhs, {2}, Core::DataType::Float64, K_SECOND)},
                 {"solution", replace<GmresCall>(&GmresCall::solution, {2}, Core::DataType::Float64, K_THIRD)},
                 {"residual", replace<GmresCall>(&GmresCall::residual, {1}, Core::DataType::Float64, K_FIFTH)},
                 {"iterations", replace<GmresCall>(&GmresCall::iterations, {1}, Core::DataType::Float32, K_FOURTH)}});
        }

        /** @brief Tests that an operand of the wrong element count, and an empty system, are refused naming the operand. */
        TEST(Small_GmresOperandCheck, AWrongElementCountIsRefusedNamingTheOperand)
        {
            expect_every_row_refused<GmresCall>(
                K_GMRES,
                {{"rhs", replace<GmresCall>(&GmresCall::rhs, {0}, Core::DataType::Float32, K_SECOND)},
                 {"matrix", replace<GmresCall>(&GmresCall::matrix, {5}, Core::DataType::Float32, K_FIRST)},
                 {"solution", replace<GmresCall>(&GmresCall::solution, {3}, Core::DataType::Float32, K_THIRD)},
                 {"iterations", replace<GmresCall>(&GmresCall::iterations, {2}, Core::DataType::UInt32, K_FOURTH)},
                 {"residual", replace<GmresCall>(&GmresCall::residual, {2}, Core::DataType::Float32, K_FIFTH)}});
        }

        /** @brief Tests that an operand without storage is refused naming it. */
        TEST(Small_GmresOperandCheck, AnOperandWithoutStorageIsRefusedNamingIt)
        {
            expect_missing_storage_refused<GmresCall>(
                K_GMRES, {{"matrix", &GmresCall::matrix},
                          {"rhs", &GmresCall::rhs},
                          {"solution", &GmresCall::solution},
                          {"iterations", &GmresCall::iterations},
                          {"residual", &GmresCall::residual}});
        }

        /** @brief Tests that a strided view is refused naming the operand. */
        TEST(Small_GmresOperandCheck, AStridedViewIsRefusedNamingIt)
        {
            expect_strided_view_refused<GmresCall>(
                K_GMRES, {{"matrix", &GmresCall::matrix},
                          {"rhs", &GmresCall::rhs},
                          {"solution", &GmresCall::solution}});
        }

        /** @brief Tests that an operand on a device without fp64 is refused naming it. */
        TEST(Small_GmresOperandCheck, AnOperandOnADeviceWithoutFp64IsRefusedNamingIt)
        {
            expect_wide_operand_on_narrow_device_refused<GmresCall>(
                K_GMRES, {{"matrix", &GmresCall::matrix},
                          {"rhs", &GmresCall::rhs},
                          {"solution", &GmresCall::solution},
                          {"residual", &GmresCall::residual}});
        }

        /** @brief Tests that a solution that shares a byte with the matrix or the right-hand side is refused naming it. */
        TEST(Small_GmresOperandCheck, ASolutionThatSharesBytesWithAnInputIsRefusedNamingIt)
        {
            expect_overlap_refused<GmresCall>(K_GMRES, "solution", &GmresCall::solution,
                                              &GmresCall::matrix);
            expect_overlap_refused<GmresCall>(K_GMRES, "solution", &GmresCall::solution,
                                              &GmresCall::rhs);
        }
    } // namespace Test
} // namespace SushiBLAS
