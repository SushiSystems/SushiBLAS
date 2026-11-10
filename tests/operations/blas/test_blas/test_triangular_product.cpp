/**************************************************************************/
/* test_triangular_product.cpp                                            */
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
 * @file test_triangular_product.cpp
 * @brief Tests the parallel TRMV kernel and the public TRMV against exact integer products, a
 *        double-precision truth and each other.
 * @author Mustafa Garip
 */

#include <string>
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

        namespace
        {
            /** @brief Names the three ways a TRMV runs: two kernels of the public call and one row shape. */
            enum class Route
            {
                Reference,
                Sycl,
                SyclWorkGroup
            };

            /** @brief Holds the flags, the order and the increment of x of one exact product. */
            struct ProductShape
            {
                TriangularFlags flags; /**< The flags of the call. */
                int64_t order;         /**< The order of A. */
                int64_t increment;     /**< The distance between two elements of x. */
            };

            /** @brief Holds the flags of one bounded product and the route it runs by. */
            struct FlagsRoute
            {
                TriangularFlags flags; /**< The flags of the call. */
                Route route;           /**< The route the product runs by. */
            };

            /** @brief Returns the kernel a route selects in the public call. */
            [[nodiscard]] ExecutionMode mode_of(Route route) noexcept
            {
                return route == Route::Reference ? ExecutionMode::ForceReference
                                                 : ExecutionMode::ForceSYCL;
            }

            /** @brief Returns the tail of a row named by a route. */
            [[nodiscard]] std::string route_name(Route route)
            {
                if (route == Route::Reference)
                    return "Reference";
                return route == Route::Sycl ? "Sycl" : "SyclWorkGroup";
            }

            /** @brief Returns the tail of a row named by a shape of product. */
            [[nodiscard]] std::string axis_name(const ProductShape& shape)
            {
                return TriangularSupport::axis_name(shape.flags) + "_n" +
                       std::to_string(shape.order) + "_inc" + std::to_string(shape.increment);
            }

            /** @brief Returns the tail of a row named by flags and a route. */
            [[nodiscard]] std::string axis_name(const FlagsRoute& axis)
            {
                return TriangularSupport::axis_name(axis.flags) + "_" + route_name(axis.route);
            }

            /** @brief Returns the rows of the exact product: every flag set, order and increment. */
            [[nodiscard]] std::vector<TriangularRow<ProductShape>> product_rows()
            {
                std::vector<TriangularRow<ProductShape>> rows;
                for (const TriangularRow<TriangularFlags>& row :
                     flag_rows(K_TRIANGULAR_TYPES, {Core::Side::Left}))
                    for (const int64_t order : K_PRODUCT_ORDERS)
                        for (const int64_t increment : K_INCREMENTS)
                            rows.push_back({row.type, ProductShape{row.axis, order, increment}});
                return rows;
            }

            /** @brief Returns the rows of the bounded product: each triangle, form and route. */
            [[nodiscard]] std::vector<TriangularRow<FlagsRoute>> bound_rows()
            {
                std::vector<TriangularRow<FlagsRoute>> rows;
                for (const Core::DataType type : K_TRIANGULAR_REAL_TYPES)
                    for (const Core::Uplo uplo : {Core::Uplo::Lower, Core::Uplo::Upper})
                        for (const Core::Transpose trans :
                             {Core::Transpose::NoTrans, Core::Transpose::Trans})
                            for (const Route route :
                                 {Route::Reference, Route::Sycl, Route::SyclWorkGroup})
                                rows.push_back(
                                    {type, FlagsRoute{plain_flags(uplo, trans), route}});
                return rows;
            }
        } // namespace

        /** @brief Runs the exact products, one shape per row. */
        class Medium_TriangularProduct : public TriangularCase<ProductShape>
        {
            protected:
                /**
                 * @brief Expects the product of x, increment apart, to equal the host product by
                 *        both kernels of the public call and both row shapes of the SYCL kernel.
                 */
                template<typename T>
                void expect_exact_on_every_route(const HostSolve<T>& system,
                                                 const std::vector<T>& x, int64_t increment)
                {
                    const T gap = element_of<T>(K_SENTINEL);
                    const std::vector<T> input = spaced(x, increment, gap);
                    const std::vector<T> expected = spaced(system.product(x), increment, gap);

                    for (const ExecutionMode mode : K_MODES)
                    {
                        SCOPED_TRACE(word(mode));
                        expect_equal_values(multiply_through_view(system, input, increment, mode),
                                            expected);
                    }

                    for (const std::size_t group_width : K_GROUP_WIDTHS)
                    {
                        SCOPED_TRACE(::testing::Message() << "group width " << group_width);
                        expect_equal_values(
                            multiply_with_kernel(system, input, increment, group_width), expected);
                    }
                }
        };

        /** @brief Runs the products that take no axis besides the element type. */
        class Medium_TriangularProductOnly : public TriangularCase<NoAxis>
        {

        };

        /** @brief Runs the products measured against the truth, one route per row. */
        class Medium_TriangularProductBound : public TriangularCase<FlagsRoute>
        {
            protected:
                /** @brief Returns the product of x by the route of the row. */
                template<typename T>
                [[nodiscard]] std::vector<T> multiply_by(const HostSolve<T>& system,
                                                         const std::vector<T>& x, Route route)
                {
                    if (route == Route::SyclWorkGroup)
                        return multiply_with_kernel(system, x, 1, K_GROUP_WIDTH);
                    return multiply_through_view(system, x, 1, mode_of(route));
                }
        };

        /** @brief Tests that TRMV of small integers equals the host product by every route. */
        TEST_P(Medium_TriangularProduct, TrmvIsExactOnSmallIntegers)
        {
            with_non_half_element(GetParam().type, [&](auto element) {
                using T = typename decltype(element)::type;
                Draws draws(K_SEED);
                const ProductShape& shape = GetParam().axis;
                const std::size_t elements = static_cast<std::size_t>(shape.order * shape.order);

                const HostSolve<T> system(shape.flags, shape.order, 1,
                                          integer_values<T>(elements, -3, 3, draws));
                const std::vector<T> x =
                    integer_values<T>(static_cast<std::size_t>(shape.order), -3, 3, draws);

                expect_exact_on_every_route(system, x, shape.increment);
            });
        }

        /** @brief Tests that two runs of one TRMV on the SYCL kernel leave the same bytes. */
        TEST_P(Medium_TriangularProductOnly, TrmvRepeatsToTheBit)
        {
            with_non_half_element(GetParam().type, [&](auto element) {
                using T = typename decltype(element)::type;
                Draws draws(K_SEED);
                const std::size_t order = static_cast<std::size_t>(K_PRODUCT_BOUND_ORDER);

                const HostSolve<T> system(TriangularFlags{}, K_PRODUCT_BOUND_ORDER, 1,
                                          fraction_values<T>(order * order, draws));
                const std::vector<T> x = fraction_values<T>(order, draws);

                expect_same_bits(multiply_through_view(system, x, 1, ExecutionMode::ForceSYCL),
                                 multiply_through_view(system, x, 1, ExecutionMode::ForceSYCL));
            });
        }

        /** @brief Tests that the snapshot of a TRMV returns to the pool and is lent again. */
        TEST_P(Medium_TriangularProductOnly, TrmvReturnsItsSnapshotToThePool)
        {
            with_non_half_element(GetParam().type, [&](auto element) {
                using T = typename decltype(element)::type;
                Draws draws(K_SEED);
                const std::size_t order = static_cast<std::size_t>(K_POOLED_ORDER);

                const HostSolve<T> system(TriangularFlags{}, K_POOLED_ORDER, 1,
                                          fraction_values<T>(order * order, draws));
                Tensor a_buffer = device_buffer<T>(*engine, system.stored_matrix());
                Tensor x = device_buffer<T>(*engine, fraction_values<T>(order, draws));
                Tensor A = matrix_view(a_buffer, {K_POOLED_ORDER, K_POOLED_ORDER},
                                       Core::Layout::RowMajor);

                ScratchPool& pool = engine->scratch_pool();
                set_mode(ExecutionMode::ForceSYCL);
                const std::size_t outstanding_before =
                    TestEnvironment::live_allocation_count() - pool.idle_block_count();

                SushiBLAS::BLASOps(recorder()).trmv(A, x, Core::Uplo::Lower);
                recording->run().value();

                EXPECT_EQ(outstanding_before,
                          TestEnvironment::live_allocation_count() - pool.idle_block_count())
                    << "Every allocation the product made must be idle in the pool or freed.";

                const std::size_t live_after_first = TestEnvironment::live_allocation_count();
                const std::size_t idle_after_first = pool.idle_block_count();

                SushiBLAS::BLASOps(recorder()).trmv(A, x, Core::Uplo::Lower);
                recording->run().value();

                EXPECT_EQ(live_after_first, TestEnvironment::live_allocation_count());
                EXPECT_EQ(idle_after_first, pool.idle_block_count());
            });
        }

        /** @brief Tests that both launches of TRMV stay behind the events the kernel is given. */
        TEST_P(Medium_TriangularProductOnly, TrmvWaitsForItsDependencies)
        {
            with_non_half_element(GetParam().type, [&](auto element) {
                using T = typename decltype(element)::type;
                Draws draws(K_SEED);
                const std::size_t order = static_cast<std::size_t>(K_ORDER);

                const HostSolve<T> system(TriangularFlags{}, K_ORDER, 1,
                                          integer_values<T>(order * order, -3, 3, draws));
                const std::vector<T> x = integer_values<T>(order, -3, 3, draws);

                Tensor A = device_buffer<T>(*engine, system.stored_matrix());
                Tensor device_x = device_buffer<T>(*engine, x);
                Tensor work =
                    device_buffer<T>(*engine, std::vector<T>(order, element_of<T>(0.0)));

                const TriangularFlags& flags = system.flags();
                DependencyGate gate(test_queue());
                sycl::event done = Kernels::SYCL::trmv<T>(
                    test_queue(), flags.layout, flags.uplo, flags.trans, flags.diag,
                    system.order(), pointer_of<const T>(A), system.lda(),
                    pointer_of<T>(device_x), 1, pointer_of<T>(work),
                    std::vector<sycl::event>{gate.event()});

                EXPECT_FALSE(is_complete(done)) << "The product's last event completed while its "
                                                   "dependency was still running.";

                gate.open();
                done.wait_and_throw();
                expect_equal_values(read_buffer<T>(device_x), system.product(x));
            });
        }

        /** @brief Tests that TRMV stays within its rounding bound of the truth by one route. */
        TEST_P(Medium_TriangularProductBound, TrmvIsWithinItsBoundOfTheTruth)
        {
            with_single_or_double_element(GetParam().type, [&](auto element) {
                using T = typename decltype(element)::type;
                Draws draws(K_SEED);
                const std::size_t order = static_cast<std::size_t>(K_PRODUCT_BOUND_ORDER);

                const HostSolve<T> system(GetParam().axis.flags, K_PRODUCT_BOUND_ORDER, 1,
                                          fraction_values<T>(order * order, draws));
                const std::vector<T> x = fraction_values<T>(order, draws);

                expect_product_within_bound(system, x,
                                            multiply_by(system, x, GetParam().axis.route),
                                            mode_of(GetParam().axis.route));
            });
        }

        INSTANTIATE_TEST_SUITE_P(, Medium_TriangularProduct, ::testing::ValuesIn(product_rows()),
                                 triangular_row_name<ProductShape>);
        INSTANTIATE_TEST_SUITE_P(, Medium_TriangularProductOnly,
                                 ::testing::ValuesIn(plain_rows_over(K_TRIANGULAR_TYPES)),
                                 triangular_row_name<NoAxis>);
        INSTANTIATE_TEST_SUITE_P(, Medium_TriangularProductBound,
                                 ::testing::ValuesIn(bound_rows()),
                                 triangular_row_name<FlagsRoute>);
    } // namespace Test
} // namespace SushiBLAS
