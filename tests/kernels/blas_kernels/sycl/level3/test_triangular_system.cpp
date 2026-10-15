/**************************************************************************/
/* test_triangular_system.cpp                                             */
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
 * @file test_triangular_system.cpp
 * @brief Tests the decoding of a TRSM call into the steps and vectors of one substitution.
 * @author Mustafa Garip
 */

#include <array>
#include <vector>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <SushiBLAS/blas_kernels/sycl/level3/triangular_system.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/core/blas_types.hpp>
#include <SushiBLAS/blas_seam_kernels/operand_access.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief Holds the order of A in the decoding cases.
            constexpr int64_t K_ORDER = 4;

            /// @brief Holds the right-hand-side vectors of B in the decoding cases.
            constexpr int64_t K_VECTORS = 3;

            /// @brief Holds the leading dimension of A in the decoding cases.
            constexpr int64_t K_LDA = 5;

            /// @brief Holds the leading dimension of B in the decoding cases.
            constexpr int64_t K_LDB = 6;

            /** @brief Returns the rows of B: the order on the left, else the vectors. */
            [[nodiscard]] constexpr int64_t rows_of(Core::Side side) noexcept
            {
                return side == Core::Side::Left ? K_ORDER : K_VECTORS;
            }

            /** @brief Returns the columns of B: the vectors on the left, else the order. */
            [[nodiscard]] constexpr int64_t cols_of(Core::Side side) noexcept
            {
                return side == Core::Side::Left ? K_VECTORS : K_ORDER;
            }

            /// @brief Holds the two layouts every decoding case runs in.
            constexpr Core::Layout K_LAYOUTS[] = {Core::Layout::RowMajor,
                                                  Core::Layout::ColumnMajor};

            /// @brief Holds the two sides every decoding case runs on.
            constexpr Core::Side K_SIDES[] = {Core::Side::Left, Core::Side::Right};

            /** @brief Holds the flags of one TRSM call. */
            struct SolveFlags
            {
                Core::Layout layout;   /**< The memory order of A and B. */
                Core::Side side;       /**< The side op(A) stands on. */
                Core::Uplo uplo;       /**< The triangle of A that is read. */
                Core::Transpose trans; /**< The op applied to A. */
                Core::Diag diag;       /**< Whether the diagonal of A is one. */
            };

            /** @brief Returns every combination of layout, side, triangle, op and diagonal. */
            [[nodiscard]] std::vector<SolveFlags> every_solve_flags()
            {
                std::vector<SolveFlags> all;
                for (const Core::Layout layout : K_LAYOUTS)
                    for (const Core::Side side : K_SIDES)
                        for (const Core::Uplo uplo : {Core::Uplo::Upper, Core::Uplo::Lower})
                            for (const Core::Transpose trans :
                                 {Core::Transpose::NoTrans, Core::Transpose::Trans,
                                  Core::Transpose::ConjTrans})
                                for (const Core::Diag diag :
                                     {Core::Diag::NonUnit, Core::Diag::Unit})
                                    all.push_back(SolveFlags{layout, side, uplo, trans, diag});
                return all;
            }
        } // namespace

        /**
         * @brief Tests that every side and flag of TRSM decodes to the substitution of its row
         *        in DESIGN_BLAS_L3, section 7.4.
         */
        TEST(Unit_TriangularSystem, EverySideAndFlagDecodesToItsLoop)
        {
            std::array<std::complex<float>, K_LDA * K_LDA> a;
            for (std::size_t i = 0; i < a.size(); ++i)
                a[i] = std::complex<float>(static_cast<float>(i + 1),
                                           -static_cast<float>(i + 1) - 0.5f);

            for (const SolveFlags& flags : every_solve_flags())
            {
                const Core::Layout layout = flags.layout;
                const Core::Side side = flags.side;
                const Core::Uplo uplo = flags.uplo;
                const Core::Transpose trans = flags.trans;
                const Core::Diag diag = flags.diag;

                SCOPED_TRACE(::testing::Message()
                             << (layout == Core::Layout::RowMajor ? "ROW_MAJOR" : "COLUMN_MAJOR")
                             << " " << Core::to_string(side) << " " << Core::to_string(uplo)
                             << " " << Core::to_string(trans) << " " << Core::to_string(diag));

                const bool left = (side == Core::Side::Left);
                const Kernels::SYCL::TriangularSystem system = Kernels::SYCL::triangular_system(
                    layout, side, uplo, trans, diag, rows_of(side), cols_of(side), K_LDA, K_LDB);

                const Kernels::OperandAccess op_a = Kernels::operand_access(layout, trans, K_LDA);
                const Kernels::OperandAccess at_b =
                    Kernels::operand_access(layout, Core::Transpose::NoTrans, K_LDB);

                EXPECT_EQ(K_ORDER, system.order);
                EXPECT_EQ(K_VECTORS, system.vectors);
                EXPECT_EQ(diag == Core::Diag::Unit, system.unit_diag);

                // op(A) is upper when exactly one of an upper A and a transposing op holds.
                const bool op_upper =
                    (uplo == Core::Uplo::Upper) != (trans != Core::Transpose::NoTrans);
                EXPECT_EQ(left ? !op_upper : op_upper, system.forward);

                for (int64_t s = 0; s < K_ORDER; ++s)
                {
                    for (int64_t t = 0; t < K_ORDER; ++t)
                        EXPECT_EQ(left ? op_a.at(a.data(), s, t) : op_a.at(a.data(), t, s),
                                  system.coefficient.at(a.data(), s, t))
                            << "step " << s << ", unknown " << t;

                    for (int64_t v = 0; v < K_VECTORS; ++v)
                        EXPECT_EQ(left ? at_b.index(s, v) : at_b.index(v, s),
                                  system.unknown.index(s, v))
                            << "step " << s << ", vector " << v;
                }
            }
        }

        /**
         * @brief Tests that CONJ_TRANS conjugates the factors read from A and no element of B.
         */
        TEST(Unit_TriangularSystem, ConjTransConjugatesTheCoefficientAndNotTheUnknown)
        {
            for (const Core::Layout layout : K_LAYOUTS)
            {
                for (const Core::Side side : K_SIDES)
                {
                    const Kernels::SYCL::TriangularSystem conjugating =
                        Kernels::SYCL::triangular_system(
                            layout, side, Core::Uplo::Lower, Core::Transpose::ConjTrans,
                            Core::Diag::NonUnit, rows_of(side), cols_of(side), K_LDA, K_LDB);
                    EXPECT_TRUE(conjugating.coefficient.conjugated);
                    EXPECT_FALSE(conjugating.unknown.conjugated);

                    const Kernels::SYCL::TriangularSystem transposing =
                        Kernels::SYCL::triangular_system(
                            layout, side, Core::Uplo::Lower, Core::Transpose::Trans,
                            Core::Diag::NonUnit, rows_of(side), cols_of(side), K_LDA, K_LDB);
                    EXPECT_FALSE(transposing.coefficient.conjugated);
                    EXPECT_FALSE(transposing.unknown.conjugated);
                }
            }
        }
    } // namespace Test
} // namespace SushiBLAS
