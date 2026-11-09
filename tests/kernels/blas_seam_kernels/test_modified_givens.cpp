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

            /// @brief The outcome of a swap (flag 1): equal products, so H is [[1, 1], [-1, 1]].
            const ModifiedGivensOutcome K_SWAP = {
                1.0, 1.0, 1.0, 1.0, {1.0, 1.0, K_UNWRITTEN, K_UNWRITTEN, 1.0}, 0.5, 0.5, 2.0};

            /// @brief The outcome of a plain rotation (flag 0): the first product is larger.
            const ModifiedGivensOutcome K_KEEP = {
                5.0, 5.0, 4.0, 2.0, {0.0, K_UNWRITTEN, -0.5, 0.5, K_UNWRITTEN}, 4.0, 4.0, 5.0};

            /// @brief The outcome of the identity (flag -2): the second product is zero.
            const ModifiedGivensOutcome K_NONE = {
                3.0, 0.0, 5.0, 7.0,
                {-2.0, K_UNWRITTEN, K_UNWRITTEN, K_UNWRITTEN, K_UNWRITTEN}, 3.0, 0.0, 5.0};

            /// @brief The outcome of a refusal (flag -1 with zeros): d1 is negative.
            const ModifiedGivensOutcome K_REFUSED = {
                -1.0, 1.0, 1.0, 1.0, {-1.0, 0.0, 0.0, 0.0, 0.0}, 0.0, 0.0, 0.0};

            /**
             * @brief Expects that modified_givens and store_modified_givens give an outcome:
             *        the stored array and the three returned values.
             */
            template<typename R>
            void expect_outcome(const ModifiedGivensOutcome& outcome)
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
        } // namespace

        /**
         * @brief Names the suite of the modified Givens cases of component type R.
         */
        template<typename R>
        class Small_ModifiedGivens : public ::testing::Test
        {

        };

        /// @brief Names the real types the modified rotation suite runs over.
        using ModifiedGivensTypes = ::testing::Types<float, double>;
        SB_TYPED_TEST_SUITE(Small_ModifiedGivens, ModifiedGivensTypes);

        /** @brief Tests that a swap stores its flag, h11, h22 and the scaled values. */
        TYPED_TEST(Small_ModifiedGivens, ASwapStoresOnlyWhatItsFlagDoesNotImply)
        {
            expect_outcome<TypeParam>(K_SWAP);
        }

        /** @brief Tests that a plain rotation stores its flag, h21 and h12. */
        TYPED_TEST(Small_ModifiedGivens, ARotationStoresOnlyWhatItsFlagDoesNotImply)
        {
            expect_outcome<TypeParam>(K_KEEP);
        }

        /** @brief Tests that a zero second product stores the identity flag and nothing else. */
        TYPED_TEST(Small_ModifiedGivens, TheIdentityStoresOnlyItsFlag)
        {
            expect_outcome<TypeParam>(K_NONE);
        }

        /** @brief Tests that a negative first factor stores the refusal flag and zeros. */
        TYPED_TEST(Small_ModifiedGivens, ARefusalStoresItsFlagAndZeros)
        {
            expect_outcome<TypeParam>(K_REFUSED);
        }

        /**
         * @brief Tests that scale factors above the window are divided back into it.
         */
        TYPED_TEST(Small_ModifiedGivens, LargeFactorsAreRescaledDown)
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
        TYPED_TEST(Small_ModifiedGivens, SmallFactorsAreRescaledUp)
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
         * @brief Tests that a first factor no pass brings into the window ends the rescaling
         *        at the pass limit: d1 stays infinite, and x1 and h11 grow to infinity.
         */
        TYPED_TEST(Small_ModifiedGivens, AnInfiniteFirstFactorEndsTheRescalingAtThePassLimit)
        {
            using R = TypeParam;

            const Kernels::ModifiedGivens<R> h = Kernels::modified_givens(
                std::numeric_limits<R>::infinity(), R(1), R(1), R(1));

            EXPECT_EQ(R(-1), h.flag);
            EXPECT_TRUE(std::isinf(h.d1));
            EXPECT_TRUE(std::isinf(h.x1));
            EXPECT_TRUE(std::isinf(h.h11));
            EXPECT_EQ(R(1), h.d2);
        }
    } // namespace Test
} // namespace SushiBLAS
