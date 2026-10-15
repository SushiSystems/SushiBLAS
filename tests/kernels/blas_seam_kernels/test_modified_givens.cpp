/**************************************************************************/
/* test_modified_givens.cpp                                               */
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
 * @file test_modified_givens.cpp
 * @brief Tests the modified Givens transformation brick and the store of its parameters.
 * @author Mustafa Garip
 */

#include <cmath>
#include <limits>
#include <cstdint>
#include <gtest/gtest.h>
#include <SushiBLAS/blas_seam_kernels/modified_givens.hpp>
#include <SushiBLAS/typed_suite_naming/default_type_names.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief Holds the value store_modified_givens must leave in an implied slot.
            constexpr double K_UNWRITTEN = 99.0;

            /// @brief Holds the factor xROTMG rescales by, written here without the brick.
            constexpr double K_GAMMA = 4096.0;

            /// @brief Holds the square of K_GAMMA, the top of the xROTMG scale window.
            constexpr double K_GAMMA_SQUARED = 16777216.0;

            /** @brief Holds one call of modified_givens and what it must leave. */
            struct ModifiedGivensOutcome
            {
                double d1;       /**< The first scale factor passed in. */
                double d2;       /**< The second scale factor passed in. */
                double x1;       /**< The first component passed in. */
                double y1;       /**< The second component passed in. */
                double param[5]; /**< The parameter array expected after the store. */
                double d1_after; /**< The expected first scale factor. */
                double d2_after; /**< The expected second scale factor. */
                double x1_after; /**< The expected first component. */
            };

            /**
             * @brief Returns whether modified_givens gives the four values worked by hand from
             *        the reference routine.
             * @tparam R float or double; every value below is exact in both.
             */
            template<typename R>
            [[nodiscard]] constexpr bool modified_givens_matches_hand_values()
            {
                const Kernels::ModifiedGivens<R> swap =
                    Kernels::modified_givens(R(1), R(1), R(1), R(1));
                if (swap.flag != R(1) || swap.h11 != R(1) || swap.h22 != R(1) ||
                    swap.d1 != R(0.5) || swap.d2 != R(0.5) || swap.x1 != R(2))
                    return false;

                const Kernels::ModifiedGivens<R> keep =
                    Kernels::modified_givens(R(5), R(5), R(4), R(2));
                if (keep.flag != R(0) || keep.h21 != R(-0.5) || keep.h12 != R(0.5) ||
                    keep.d1 != R(4) || keep.d2 != R(4) || keep.x1 != R(5))
                    return false;

                const Kernels::ModifiedGivens<R> none =
                    Kernels::modified_givens(R(3), R(0), R(5), R(7));
                if (none.flag != R(-2) || none.d1 != R(3) || none.d2 != R(0) || none.x1 != R(5))
                    return false;

                const Kernels::ModifiedGivens<R> refused =
                    Kernels::modified_givens(R(-1), R(1), R(1), R(1));
                return refused.flag == R(-1) && refused.d1 == R(0) && refused.x1 == R(0);
            }
        } // namespace

        /**
         * @brief Names the suite of the modified Givens cases of component type R.
         */
        template<typename R>
        class Unit_ModifiedGivens : public ::testing::Test
        {

        };

        /// @brief Names the real types the modified rotation suite runs over.
        using ModifiedGivensTypes = ::testing::Types<float, double>;
        SB_TYPED_TEST_SUITE(Unit_ModifiedGivens, ModifiedGivensTypes);

        /**
         * @brief Tests at compile time the four hand values of the reference routine.
         */
        TYPED_TEST(Unit_ModifiedGivens, TheHandValuesHoldAtCompileTime)
        {
            static_assert(modified_givens_matches_hand_values<TypeParam>(),
                          "modified_givens gives the values worked by hand.");

            SUCCEED();
        }

        /**
         * @brief Tests the four outcomes of xROTMG and which parameters each one stores.
         */
        TYPED_TEST(Unit_ModifiedGivens, EachOutcomeStoresOnlyWhatItsFlagDoesNotImply)
        {
            using R = TypeParam;
            constexpr double unwritten = K_UNWRITTEN;

            const ModifiedGivensOutcome outcomes[] = {
                {1.0, 1.0, 1.0, 1.0, {1.0, 1.0, unwritten, unwritten, 1.0}, 0.5, 0.5, 2.0},
                {5.0, 5.0, 4.0, 2.0, {0.0, unwritten, -0.5, 0.5, unwritten}, 4.0, 4.0, 5.0},
                {3.0, 0.0, 5.0, 7.0, {-2.0, unwritten, unwritten, unwritten, unwritten},
                 3.0, 0.0, 5.0},
                {-1.0, 1.0, 1.0, 1.0, {-1.0, 0.0, 0.0, 0.0, 0.0}, 0.0, 0.0, 0.0}};

            for (const ModifiedGivensOutcome& outcome : outcomes)
            {
                SCOPED_TRACE(::testing::Message()
                             << "d1 " << outcome.d1 << ", d2 " << outcome.d2 << ", x1 "
                             << outcome.x1 << ", y1 " << outcome.y1);

                const Kernels::ModifiedGivens<R> h = Kernels::modified_givens(
                    static_cast<R>(outcome.d1), static_cast<R>(outcome.d2),
                    static_cast<R>(outcome.x1), static_cast<R>(outcome.y1));

                R param[Kernels::K_MODIFIED_GIVENS_PARAMETERS];
                for (R& slot : param)
                    slot = static_cast<R>(K_UNWRITTEN);
                Kernels::store_modified_givens(h, param);

                for (int slot = 0; slot < Kernels::K_MODIFIED_GIVENS_PARAMETERS; ++slot)
                    EXPECT_EQ(static_cast<R>(outcome.param[slot]), param[slot])
                        << "param[" << slot << "]";

                EXPECT_EQ(static_cast<R>(outcome.d1_after), h.d1);
                EXPECT_EQ(static_cast<R>(outcome.d2_after), h.d2);
                EXPECT_EQ(static_cast<R>(outcome.x1_after), h.x1);
            }
        }

        /**
         * @brief Tests that scale factors above the window are divided back into it.
         */
        TYPED_TEST(Unit_ModifiedGivens, LargeFactorsAreRescaledDown)
        {
            using R = TypeParam;
            const R gamma = static_cast<R>(K_GAMMA);
            const R factor = std::ldexp(R(1), 26);

            const Kernels::ModifiedGivens<R> h =
                Kernels::modified_givens(factor, factor, R(4), R(2));

            EXPECT_EQ(R(-1), h.flag);
            EXPECT_EQ(gamma, h.h11);
            EXPECT_EQ(gamma, h.h22);
            EXPECT_EQ(gamma / R(2), h.h12);
            EXPECT_EQ(-gamma / R(2), h.h21);
            EXPECT_EQ(R(5) * gamma, h.x1);

            const R scaled = (factor / R(1.25)) / static_cast<R>(K_GAMMA_SQUARED);
            EXPECT_EQ(scaled, h.d1);
            EXPECT_EQ(scaled, h.d2);
        }

        /**
         * @brief Tests that scale factors below the window are multiplied back into it.
         */
        TYPED_TEST(Unit_ModifiedGivens, SmallFactorsAreRescaledUp)
        {
            using R = TypeParam;
            const R gamma = static_cast<R>(K_GAMMA);
            const R factor = std::ldexp(R(1), -30);

            const Kernels::ModifiedGivens<R> h =
                Kernels::modified_givens(factor, factor, R(4), R(2));

            EXPECT_EQ(R(-1), h.flag);
            EXPECT_EQ(R(1) / gamma, h.h11);
            EXPECT_EQ(R(1) / gamma, h.h22);
            EXPECT_EQ(R(0.5) / gamma, h.h12);
            EXPECT_EQ(R(-0.5) / gamma, h.h21);
            EXPECT_EQ(R(5) / gamma, h.x1);

            const R scaled = (factor / R(1.25)) * static_cast<R>(K_GAMMA_SQUARED);
            EXPECT_EQ(scaled, h.d1);
            EXPECT_EQ(scaled, h.d2);
        }

        /**
         * @brief Tests that a scale factor no pass brings into the window ends the rescaling.
         */
        TYPED_TEST(Unit_ModifiedGivens, AnInfiniteFactorReturns)
        {
            using R = TypeParam;

            const Kernels::ModifiedGivens<R> h = Kernels::modified_givens(
                std::numeric_limits<R>::infinity(), R(1), R(1), R(1));

            EXPECT_EQ(R(-1), h.flag);
            EXPECT_TRUE(std::isinf(h.d1));
            EXPECT_EQ(R(1), h.d2);
        }
    } // namespace Test
} // namespace SushiBLAS
