/**************************************************************************/
/* test_triangular_solve.cpp                                              */
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
 * @file test_triangular_solve.cpp
 * @brief Tests the blocked TRSM kernel and the public TRSV and TRSM against exact integer
 *        systems, a residual bound and each other.
 * @author Mustafa Garip
 */

#include <string>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include "triangular_support.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        using namespace TriangularSupport;

        namespace
        {
            /** @brief Holds the side of a batched solve and whether its matrices share one A. */
            struct BatchForm
            {
                Core::Side side; /**< The side of op(A). */
                bool shares_a;   /**< Whether every matrix of the batch uses the first A. */
            };

            /** @brief Names the two systems that hold no element to solve. */
            enum class EmptyForm
            {
                NoSteps,
                NoVectors
            };

            /** @brief Holds the flags of a call and the kernel it runs in. */
            struct FlagsMode
            {
                TriangularFlags flags; /**< The flags of the call. */
                ExecutionMode mode;    /**< The kernel the call runs in. */
            };

            /** @brief Returns the tail of a row named by a batch form. */
            [[nodiscard]] std::string axis_name(const BatchForm& form)
            {
                return "_" + word(form.side) + (form.shares_a ? "_SharedA" : "_BatchedA");
            }

            /** @brief Returns the tail of a row named by an empty system. */
            [[nodiscard]] std::string axis_name(EmptyForm form)
            {
                return form == EmptyForm::NoSteps ? "_NoSteps" : "_NoVectors";
            }

            /** @brief Returns the tail of a row named by flags and a kernel. */
            [[nodiscard]] std::string axis_name(const FlagsMode& axis)
            {
                return TriangularSupport::axis_name(axis.flags) + "_" + word(axis.mode);
            }

            /** @brief Returns the rows of the batched solve: both sides, shared or batched A. */
            [[nodiscard]] std::vector<TriangularRow<BatchForm>> batch_rows()
            {
                std::vector<BatchForm> forms;
                for (const Core::Side side : {Core::Side::Left, Core::Side::Right})
                    for (const bool shares_a : {false, true})
                        forms.push_back({side, shares_a});
                return rows_over(K_TRIANGULAR_TYPES, forms);
            }

            /** @brief Returns the rows of the residual bound: every flag set in both kernels. */
            [[nodiscard]] std::vector<TriangularRow<FlagsMode>> residual_rows()
            {
                std::vector<TriangularRow<FlagsMode>> rows;
                for (const TriangularRow<TriangularFlags>& row :
                     flag_rows(K_TRIANGULAR_REAL_TYPES, {Core::Side::Left, Core::Side::Right}))
                    for (const ExecutionMode mode : K_MODES)
                        rows.push_back({row.type, FlagsMode{row.axis, mode}});
                return rows;
            }

            /** @brief Returns the system that holds no step, or no vector, to solve. */
            template<typename T>
            [[nodiscard]] HostSolve<T> empty_system(EmptyForm form, Draws& draws)
            {
                if (form == EmptyForm::NoSteps)
                    return HostSolve<T>(TriangularFlags{}, 0, K_EMPTY_EXTENT, {});
                return HostSolve<T>(TriangularFlags{}, K_EMPTY_EXTENT, 0,
                                    exact_matrix<T>(K_EMPTY_EXTENT, draws));
            }
        } // namespace

        /** @brief Runs the solves that take the flags of a call and a block width. */
        class Medium_TriangularSolveWidth : public TriangularCase<FlagsWidth>
        {

        };

        /** @brief Runs the solves that take the flags of a call. */
        class Medium_TriangularSolveFlags : public TriangularCase<TriangularFlags>
        {

        };

        /** @brief Runs the batched solve and compares each matrix with its solve alone. */
        class Medium_TriangularSolveBatch : public TriangularCase<BatchForm>
        {
            protected:
                /**
                 * @brief Expects the matrices of a batch solved at once to equal each solved alone.
                 * @param shares_a Whether the matrices of the batch use the first A.
                 */
                template<typename T>
                void expect_batch_matches_each_alone(const SolveBatch<T>& batch, bool shares_a,
                                                     T alpha)
                {
                    const int64_t stride = shares_a ? 0 : K_ORDER * K_ORDER;
                    const std::vector<T> solved = run_solve_kernel(
                        batch.systems.front(), batch.a, stride, batch.b, K_BATCH, alpha,
                        K_NARROW_WIDTH);
                    ASSERT_EQ(solved.size(), batch.b.size());

                    const std::size_t per_matrix = batch.systems.front().elements();
                    for (std::size_t k = 0; k < batch.systems.size(); ++k)
                    {
                        SCOPED_TRACE(::testing::Message() << "matrix " << k);
                        const std::vector<T> alone = solve_with_kernel(
                            batch.systems[k],
                            slice_of(batch.b, k * per_matrix, (k + 1) * per_matrix), alpha,
                            K_ORDER);
                        expect_same_bits(slice_of(solved, k * per_matrix, (k + 1) * per_matrix),
                                         alone);
                    }
                }
        };

        /** @brief Runs the gated solve, which takes no axis besides the element type. */
        class Medium_TriangularSolve : public TriangularCase<NoAxis>
        {

        };

        /** @brief Runs the solves of a system that holds no element. */
        class Medium_TriangularSolveEmpty : public TriangularCase<EmptyForm>
        {

        };

        /** @brief Runs the solves at an order the device blocks, one triangle per row. */
        class Medium_TriangularSolveLarge : public TriangularCase<Core::Uplo>
        {

        };

        /** @brief Runs the residual bound of a solve, one kernel per row. */
        class Medium_TriangularSolveResidual : public TriangularCase<FlagsMode>
        {

        };

        /** @brief Tests that a solve leaves the same bytes at a block width as at the whole order. */
        TEST_P(Medium_TriangularSolveWidth, EveryBlockWidthGivesTheSameBits)
        {
            with_non_half_element(GetParam().type, [&](auto element) {
                using T = typename decltype(element)::type;
                Draws draws(K_SEED);
                const T alpha = element_of<T>(K_ROUNDED_ALPHA);
                const HostSolve<T> system(GetParam().axis.flags, K_ORDER, K_VECTORS,
                                          dominant_matrix<T>(K_ORDER, draws));
                const std::vector<T> b = fraction_values<T>(system.elements(), draws);

                expect_same_bits(solve_with_kernel(system, b, alpha, GetParam().axis.width),
                                 solve_with_kernel(system, b, alpha, K_ORDER));
            });
        }

        /** @brief Tests that a blocked solve of an integer system returns its solution exactly. */
        TEST_P(Medium_TriangularSolveWidth, ABlockedSolveIsExactOnIntegerSystems)
        {
            with_non_half_element(GetParam().type, [&](auto element) {
                using T = typename decltype(element)::type;
                Draws draws(K_SEED);
                const T alpha = element_of<T>(K_EXACT_ALPHA);
                const HostSolve<T> system(GetParam().axis.flags, K_ORDER, K_VECTORS,
                                          exact_matrix<T>(K_ORDER, draws));
                const std::vector<T> solution =
                    integer_values<T>(system.elements(), -3, 3, draws);

                expect_equal_values(
                    solve_with_kernel(system, system.product(solution), alpha,
                                      GetParam().axis.width),
                    scaled(solution, alpha));
            });
        }

        /** @brief Tests that the reference solve of an integer system returns its solution. */
        TEST_P(Medium_TriangularSolveFlags, TheReferenceSolveIsExactOnIntegerSystems)
        {
            with_non_half_element(GetParam().type, [&](auto element) {
                using T = typename decltype(element)::type;
                Draws draws(K_SEED);
                const HostSolve<T> system(GetParam().axis, K_ORDER, K_VECTORS,
                                          exact_matrix<T>(K_ORDER, draws));
                const std::vector<T> solution =
                    integer_values<T>(system.elements(), -3, 3, draws);

                expect_equal_values(
                    solve_through_view(system, system.product(solution), K_EXACT_ALPHA,
                                       ExecutionMode::ForceReference),
                    scaled(solution, element_of<T>(K_EXACT_ALPHA)));
            });
        }

        /** @brief Tests that a batch solved in blocks equals each of its matrices solved alone. */
        TEST_P(Medium_TriangularSolveBatch, ABatchIsSolvedMatrixByMatrix)
        {
            with_non_half_element(GetParam().type, [&](auto element) {
                using T = typename decltype(element)::type;
                Draws draws(K_SEED);
                TriangularFlags flags;
                flags.side = GetParam().axis.side;

                expect_batch_matches_each_alone(
                    make_batch<T>(flags, GetParam().axis.shares_a, draws),
                    GetParam().axis.shares_a, element_of<T>(K_ROUNDED_ALPHA));
            });
        }

        /** @brief Tests that every launch of a blocked solve stays behind the events it is given. */
        TEST_P(Medium_TriangularSolve, ABlockedSolveWaitsForItsDependencies)
        {
            with_non_half_element(GetParam().type, [&](auto element) {
                using T = typename decltype(element)::type;
                Draws draws(K_SEED);
                const T alpha = element_of<T>(K_EXACT_ALPHA);

                const HostSolve<T> system(TriangularFlags{}, K_ORDER, K_VECTORS,
                                          exact_matrix<T>(K_ORDER, draws));
                const std::vector<T> solution =
                    integer_values<T>(system.elements(), -3, 3, draws);

                Tensor A = device_buffer<T>(*engine, system.stored_matrix());
                Tensor B = device_buffer<T>(*engine, system.product(solution));

                DependencyGate gate(test_queue());
                sycl::event done = submit_solve(system, A, 0, B, 1, alpha, K_NARROW_WIDTH,
                                                std::vector<sycl::event>{gate.event()});

                EXPECT_FALSE(is_complete(done)) << "The solve's last event completed while its "
                                                   "dependency was still running.";

                gate.open();
                done.wait_and_throw();
                expect_equal_values(read_buffer<T>(B), scaled(solution, alpha));
            });
        }

        /** @brief Tests that a solve without steps, or without vectors, finishes and writes nothing. */
        TEST_P(Medium_TriangularSolveEmpty, AnEmptySystemReturnsAnEvent)
        {
            with_non_half_element(GetParam().type, [&](auto element) {
                using T = typename decltype(element)::type;
                Draws draws(K_SEED);
                const std::vector<T> untouched(static_cast<std::size_t>(K_EMPTY_EXTENT),
                                               element_of<T>(K_SENTINEL));
                const HostSolve<T> system = empty_system<T>(GetParam().axis, draws);

                expect_same_bits(run_solve_kernel(system, system.stored_matrix(), 0, untouched, 1,
                                                  element_of<T>(K_EXACT_ALPHA), 1),
                                 untouched);
            });
        }

        /**
         * @brief Tests that the public TRSV and TRSM solve a large integer system with no error
         *        on the path the device plans.
         */
        TEST_P(Medium_TriangularSolveLarge, ThePlannedPathIsExactAtALargeOrder)
        {
            with_single_or_double_element(GetParam().type, [&](auto element) {
                using T = typename decltype(element)::type;
                Draws draws(K_SEED);
                const TriangularFlags flags = plain_flags(GetParam().axis, Core::Transpose::NoTrans);

                const HostSolve<T> single(flags, K_LARGE_ORDER, 1,
                                          exact_matrix<T>(K_LARGE_ORDER, draws));
                const std::vector<T> x = integer_values<T>(single.elements(), -3, 3, draws);
                expect_equal_values(
                    solve_vector_through_view(single, single.product(x), ExecutionMode::ForceSYCL),
                    x);

                const HostSolve<T> several(flags, K_LARGE_ORDER, K_LARGE_VECTORS,
                                           exact_matrix<T>(K_LARGE_ORDER, draws));
                const std::vector<T> solution =
                    integer_values<T>(several.elements(), -3, 3, draws);
                expect_equal_values(
                    solve_through_view(several, several.product(solution), K_EXACT_ALPHA,
                                       ExecutionMode::ForceSYCL),
                    scaled(solution, element_of<T>(K_EXACT_ALPHA)));
            });
        }

        /** @brief Tests that a kernel leaves every step of a solve within its residual bound. */
        TEST_P(Medium_TriangularSolveResidual, TheResidualIsWithinItsBound)
        {
            with_single_or_double_element(GetParam().type, [&](auto element) {
                using T = typename decltype(element)::type;
                Draws draws(K_SEED);
                const HostSolve<T> system(GetParam().axis.flags, K_RESIDUAL_ORDER,
                                          K_RESIDUAL_VECTORS,
                                          dominant_matrix<T>(K_RESIDUAL_ORDER, draws));
                const std::vector<T> b = fraction_values<T>(system.elements(), draws);
                const ExecutionMode mode = GetParam().axis.mode;

                expect_residual_within_bound(
                    system, b, K_ROUNDED_ALPHA,
                    solve_through_view(system, b, K_ROUNDED_ALPHA, mode), mode);
            });
        }

        INSTANTIATE_TEST_SUITE_P(
            , Medium_TriangularSolveWidth,
            ::testing::ValuesIn(width_rows(K_TRIANGULAR_TYPES, {Core::Side::Left, Core::Side::Right},
                                           values_of(K_BLOCK_WIDTHS))),
            triangular_row_name<FlagsWidth>);
        INSTANTIATE_TEST_SUITE_P(
            , Medium_TriangularSolveFlags,
            ::testing::ValuesIn(flag_rows(K_TRIANGULAR_TYPES,
                                          {Core::Side::Left, Core::Side::Right})),
            triangular_row_name<TriangularFlags>);
        INSTANTIATE_TEST_SUITE_P(, Medium_TriangularSolveBatch, ::testing::ValuesIn(batch_rows()),
                                 triangular_row_name<BatchForm>);
        INSTANTIATE_TEST_SUITE_P(, Medium_TriangularSolve,
                                 ::testing::ValuesIn(plain_rows_over(K_TRIANGULAR_TYPES)),
                                 triangular_row_name<NoAxis>);
        INSTANTIATE_TEST_SUITE_P(
            , Medium_TriangularSolveEmpty,
            ::testing::ValuesIn(rows_over(K_TRIANGULAR_TYPES,
                                          std::vector<EmptyForm>{EmptyForm::NoSteps,
                                                                 EmptyForm::NoVectors})),
            triangular_row_name<EmptyForm>);
        INSTANTIATE_TEST_SUITE_P(
            , Medium_TriangularSolveLarge,
            ::testing::ValuesIn(rows_over(
                K_TRIANGULAR_REAL_TYPES,
                std::vector<Core::Uplo>{Core::Uplo::Lower, Core::Uplo::Upper})),
            triangular_row_name<Core::Uplo>);
        INSTANTIATE_TEST_SUITE_P(, Medium_TriangularSolveResidual,
                                 ::testing::ValuesIn(residual_rows()),
                                 triangular_row_name<FlagsMode>);
    } // namespace Test
} // namespace SushiBLAS
