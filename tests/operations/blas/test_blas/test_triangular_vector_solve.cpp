/**************************************************************************/
/* test_triangular_vector_solve.cpp                                       */
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
 * @file test_triangular_vector_solve.cpp
 * @brief Tests the blocked packed and band triangular solves, TPSV and TBSV, against exact
 *        integer systems and each other.
 * @author Mustafa Garip
 */

#include <vector>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include "triangular_support.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        using namespace TriangularSupport;

        /** @brief Runs the packed and the band solve at a block width, one flag set per row. */
        class Medium_TriangularVectorWidth : public TriangularCase<FlagsWidth>
        {
            protected:
                /**
                 * @brief Expects the packed solve, and the band solve of every band width, to
                 *        leave at one block width the bytes they leave at the whole order.
                 */
                template<typename T>
                void expect_vector_solves_agree(const TriangularFlags& flags, int64_t width,
                                                Draws& draws)
                {
                    const std::vector<T> full = dominant_matrix<T>(K_ORDER, draws);
                    const std::vector<T> b =
                        fraction_values<T>(static_cast<std::size_t>(K_ORDER), draws);

                    const HostSolve<T> packed(flags, K_ORDER, 1, full);
                    SCOPED_TRACE("packed");
                    expect_same_bits(solve_packed_with_kernel(packed, full, b, 1, width),
                                     solve_packed_with_kernel(packed, full, b, 1, K_ORDER));

                    for (const int64_t k : K_BAND_WIDTHS)
                    {
                        SCOPED_TRACE(::testing::Message() << "band of " << k);
                        const std::vector<T> narrow = banded(full, K_ORDER, k);
                        const HostSolve<T> band(flags, K_ORDER, 1, narrow);
                        expect_same_bits(solve_band_with_kernel(band, k, narrow, b, 1, width),
                                         solve_band_with_kernel(band, k, narrow, b, 1, K_ORDER));
                    }
                }
        };

        /** @brief Runs the exact packed and band solves at a block width, one flag set per row. */
        class Medium_TriangularVectorExact : public TriangularCase<FlagsWidth>
        {
            protected:
                /**
                 * @brief Expects the packed solve, and the band solve of every band width, to
                 *        return an integer solution exactly at both increments of x, and to write
                 *        nothing between the elements of x.
                 */
                template<typename T>
                void expect_vector_solves_exact(const TriangularFlags& flags, int64_t width,
                                                Draws& draws)
                {
                    const T gap = element_of<T>(K_SENTINEL);
                    const std::vector<T> full = exact_matrix<T>(K_ORDER, draws);
                    const std::vector<T> solution =
                        integer_values<T>(static_cast<std::size_t>(K_ORDER), -3, 3, draws);

                    for (const int64_t increment : K_INCREMENTS)
                    {
                        SCOPED_TRACE(::testing::Message() << "increment " << increment);
                        const std::vector<T> expected = spaced(solution, increment, gap);

                        const HostSolve<T> packed(flags, K_ORDER, 1, full);
                        SCOPED_TRACE("packed");
                        expect_equal_values(
                            solve_packed_with_kernel(
                                packed, full, spaced(packed.product(solution), increment, gap),
                                increment, width),
                            expected);

                        for (const int64_t k : K_BAND_WIDTHS)
                        {
                            SCOPED_TRACE(::testing::Message() << "band of " << k);
                            const std::vector<T> narrow = banded(full, K_ORDER, k);
                            const HostSolve<T> band(flags, K_ORDER, 1, narrow);
                            expect_equal_values(
                                solve_band_with_kernel(
                                    band, k, narrow, spaced(band.product(solution), increment, gap),
                                    increment, width),
                                expected);
                        }
                    }
                }
        };

        /** @brief Runs the vector solves that take no axis besides the element type. */
        class Medium_TriangularVectorSolve : public TriangularCase<NoAxis>
        {

        };

        /** @brief Runs the solves at an order the device blocks, one triangle per row. */
        class Medium_TriangularVectorLarge : public TriangularCase<Core::Uplo>
        {

        };

        /** @brief Tests that a packed and a band solve leave the same bytes at every block width. */
        TEST_P(Medium_TriangularVectorWidth, AVectorSolveGivesTheSameBitsAtEveryBlockWidth)
        {
            with_non_half_element(GetParam().type, [&](auto element) {
                using T = typename decltype(element)::type;
                Draws draws(K_SEED);
                expect_vector_solves_agree<T>(GetParam().axis.flags, GetParam().axis.width, draws);
            });
        }

        /**
         * @brief Tests that a blocked packed and a blocked band solve return the solution of an
         *        integer system with no error and write nothing between the elements of x.
         */
        TEST_P(Medium_TriangularVectorExact, ABlockedVectorSolveIsExactOnIntegerSystems)
        {
            with_non_half_element(GetParam().type, [&](auto element) {
                using T = typename decltype(element)::type;
                Draws draws(K_SEED);
                expect_vector_solves_exact<T>(GetParam().axis.flags, GetParam().axis.width, draws);
            });
        }

        /**
         * @brief Tests that every launch of a blocked packed solve stays behind the events it is
         *        given.
         */
        TEST_P(Medium_TriangularVectorSolve, ABlockedVectorSolveWaitsForItsDependencies)
        {
            with_non_half_element(GetParam().type, [&](auto element) {
                using T = typename decltype(element)::type;
                Draws draws(K_SEED);

                const std::vector<T> full = exact_matrix<T>(K_ORDER, draws);
                const HostSolve<T> system(TriangularFlags{}, K_ORDER, 1, full);
                const std::vector<T> solution =
                    integer_values<T>(static_cast<std::size_t>(K_ORDER), -3, 3, draws);

                Tensor AP = device_buffer<T>(*engine,
                                             packed_storage(system.flags(), K_ORDER, full));
                Tensor x = device_buffer<T>(*engine, system.product(solution));

                DependencyGate gate(test_queue());
                sycl::event done = submit_packed_solve(
                    system, AP, x, 1, K_NARROW_WIDTH, std::vector<sycl::event>{gate.event()});

                EXPECT_FALSE(is_complete(done)) << "The solve's last event completed while its "
                                                   "dependency was still running.";

                gate.open();
                done.wait_and_throw();
                expect_equal_values(read_buffer<T>(x), solution);
            });
        }

        /** @brief Tests that a packed solve of no rows finishes. */
        TEST_P(Medium_TriangularVectorSolve, AnEmptyPackedSolveReturnsAnEvent)
        {
            with_non_half_element(GetParam().type, [&](auto element) {
                using T = typename decltype(element)::type;
                const HostSolve<T> empty(TriangularFlags{}, 0, 1, {});

                EXPECT_TRUE(solve_packed_with_kernel(empty, {}, {}, 1, 1).empty());
            });
        }

        /** @brief Tests that a band solve of no rows finishes. */
        TEST_P(Medium_TriangularVectorSolve, AnEmptyBandSolveReturnsAnEvent)
        {
            with_non_half_element(GetParam().type, [&](auto element) {
                using T = typename decltype(element)::type;
                const HostSolve<T> empty(TriangularFlags{}, 0, 1, {});

                EXPECT_TRUE(solve_band_with_kernel(empty, 0, {}, {}, 1, 1).empty());
            });
        }

        /**
         * @brief Tests that the public TPSV and TBSV solve a large integer system with no error
         *        on the path the device plans.
         */
        TEST_P(Medium_TriangularVectorLarge, ThePlannedVectorPathIsExactAtALargeOrder)
        {
            with_single_or_double_element(GetParam().type, [&](auto element) {
                using T = typename decltype(element)::type;
                Draws draws(K_SEED);
                const TriangularFlags flags = plain_flags(GetParam().axis, Core::Transpose::NoTrans);

                const std::vector<T> full = exact_matrix<T>(K_LARGE_ORDER, draws);
                const std::vector<T> x =
                    integer_values<T>(static_cast<std::size_t>(K_LARGE_ORDER), -3, 3, draws);

                const HostSolve<T> packed(flags, K_LARGE_ORDER, 1, full);
                expect_equal_values(solve_packed_through_view(packed, full, packed.product(x)), x);

                const std::vector<T> narrow = banded(full, K_LARGE_ORDER, K_LARGE_BAND);
                const HostSolve<T> band(flags, K_LARGE_ORDER, 1, narrow);
                expect_equal_values(
                    solve_band_through_view(band, K_LARGE_BAND, narrow, band.product(x)), x);
            });
        }

        INSTANTIATE_TEST_SUITE_P(
            , Medium_TriangularVectorWidth,
            ::testing::ValuesIn(width_rows(K_TRIANGULAR_TYPES, {Core::Side::Left},
                                           values_of(K_BLOCK_WIDTHS))),
            triangular_row_name<FlagsWidth>);
        INSTANTIATE_TEST_SUITE_P(
            , Medium_TriangularVectorExact,
            ::testing::ValuesIn(width_rows(K_TRIANGULAR_TYPES, {Core::Side::Left},
                                           values_of(K_EXACT_WIDTHS))),
            triangular_row_name<FlagsWidth>);
        INSTANTIATE_TEST_SUITE_P(, Medium_TriangularVectorSolve,
                                 ::testing::ValuesIn(plain_rows_over(K_TRIANGULAR_TYPES)),
                                 triangular_row_name<NoAxis>);
        INSTANTIATE_TEST_SUITE_P(
            , Medium_TriangularVectorLarge,
            ::testing::ValuesIn(rows_over(
                K_TRIANGULAR_REAL_TYPES,
                std::vector<Core::Uplo>{Core::Uplo::Lower, Core::Uplo::Upper})),
            triangular_row_name<Core::Uplo>);
    } // namespace Test
} // namespace SushiBLAS
