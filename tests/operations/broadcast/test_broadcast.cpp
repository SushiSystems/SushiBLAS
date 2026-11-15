/**************************************************************************/
/* test_broadcast.cpp                                                     */
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
 * @file test_broadcast.cpp
 * @brief Tests BroadcastOps::add, sub, mul and div along each axis against a host loop, and C as A.
 * @author Mustafa Garip
 */

#include <vector>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/broadcast/broadcast.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/scalar_dtype.hpp>
#include <SushiBLAS/tensor_expectations/tensor_readback.hpp>
#include <SushiBLAS/typed_suite_naming/default_type_names.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief The forced modes every case runs in.
            constexpr ExecutionMode K_FORCED_MODES[] = {ExecutionMode::ForceReference,
                                                        ExecutionMode::ForceSYCL};

            /** @brief Names the four operations of the view. */
            enum class Arithmetic : std::uint8_t
            {
                Add,
                Sub,
                Mul,
                Div
            };

            /** @brief Names how the output of an in-place call refers to A. */
            enum class OutputAs : std::uint8_t
            {
                TheSameObject, /**< C is the very Tensor object A is. */
                AnotherView    /**< C is a copy of A, another view of the same memory. */
            };

            /** @brief Holds one matrix shape, and the shapes of the vectors that fit its prefix and suffix. */
            struct ShapeRow
            {
                /** @brief Holds the extents of A. */
                std::vector<int64_t> dims;
                /** @brief Holds the extents of the leading vector, a prefix of dims. */
                std::vector<int64_t> leading;
                /** @brief Holds the extents of the trailing vector, a suffix of dims. */
                std::vector<int64_t> trailing;
            };

            /** @brief Returns the shapes the cases run over: rank 2, 3 and 4, and tables of rank 1 to 3. */
            const std::vector<ShapeRow>& shape_rows()
            {
                static const std::vector<ShapeRow> rows = {
                    {{3, 5}, {3}, {5}},
                    {{2, 3, 4}, {2, 3}, {3, 4}},
                    {{2, 3, 4}, {2}, {4}},
                    {{2, 3, 2, 3}, {2, 3, 2}, {2, 3}},
                    {{2, 3, 2, 3}, {2}, {3}}};
                return rows;
            }

            /** @brief Returns the product of the extents. */
            std::size_t element_count(const std::vector<int64_t>& dims)
            {
                std::size_t count = 1;
                for (const int64_t extent : dims)
                    count *= static_cast<std::size_t>(extent);
                return count;
            }

            /** @brief Returns count values that are multiples of 0.25 between -0.5 and 2.0. */
            template<typename T>
            std::vector<T> quarters(std::size_t count, std::size_t seed)
            {
                constexpr std::size_t K_PERIOD = 11;
                std::vector<T> values;
                values.reserve(count);
                for (std::size_t i = 0; i < count; ++i)
                    values.push_back(static_cast<T>(
                        0.25 * (static_cast<double>((i * 5 + seed) % K_PERIOD) - 2.0)));
                return values;
            }

            /** @brief Returns count divisors from 0.25, 0.5, 1 and 2, so every quotient is exact. */
            template<typename T>
            std::vector<T> powers_of_two(std::size_t count)
            {
                std::vector<T> values;
                values.reserve(count);
                for (std::size_t i = 0; i < count; ++i)
                    values.push_back(static_cast<T>(0.25 * static_cast<double>(1u << (i % 4))));
                return values;
            }

            /** @brief Returns the vector an operation is run with. */
            template<typename T>
            std::vector<T> vector_values(Arithmetic which, std::size_t count)
            {
                return which == Arithmetic::Div ? powers_of_two<T>(count) : quarters<T>(count, 1);
            }

            /** @brief Returns the result of one operation on two values. */
            double apply(Arithmetic which, double x, double v)
            {
                switch (which)
                {
                    case Arithmetic::Add:
                        return x + v;
                    case Arithmetic::Sub:
                        return x - v;
                    case Arithmetic::Mul:
                        return x * v;
                    case Arithmetic::Div:
                        return x / v;
                }
                return 0.0;
            }

            /**
             * @brief Computes the operation on the host, element by element, in double.
             * @param v The vector; a leading one is indexed by the row of an element and a trailing
             *        one by its column.
             */
            template<typename T>
            std::vector<double> on_the_host(Arithmetic which, BroadcastAxis axis,
                                            const std::vector<T>& a, const std::vector<T>& v)
            {
                const std::size_t inner_of_leading = a.size() / v.size();
                std::vector<double> expected;
                expected.reserve(a.size());
                for (std::size_t i = 0; i < a.size(); ++i)
                {
                    const std::size_t selected =
                        axis == BroadcastAxis::Leading ? i / inner_of_leading : i % v.size();
                    expected.push_back(apply(which, static_cast<double>(a[i]),
                                             static_cast<double>(v[selected])));
                }
                return expected;
            }

            /** @brief Returns the vector widened to double. */
            template<typename T>
            std::vector<double> widened(const std::vector<T>& values)
            {
                return std::vector<double>(values.begin(), values.end());
            }

            /** @brief Returns whether the test device can compute in T. */
            template<typename T>
            bool device_computes()
            {
                const sycl::device device = TestEnvironment::queue().get_device();

                if constexpr (std::is_same_v<T, sycl::half>)
                    return device.has(sycl::aspect::fp16);
                else if constexpr (std::is_same_v<T, double>)
                    return device.has(sycl::aspect::fp64);
                else
                    return true;
            }
        } // namespace

        /** @brief Runs BroadcastOps in one element type, skipped where the device lacks it. */
        template<typename T>
        class Medium_BroadcastOpsTyped : public SushiBLASTest
        {
            protected:
                /** @brief Builds the engine and skips the case on a device without T's aspect. */
                void SetUp() override
                {
                    SushiBLASTest::SetUp();

                    if (!device_computes<T>())
                        GTEST_SKIP() << "the device does not report the aspect this dtype needs";
                }

                /** @brief Creates a tensor of the given extents holding values. */
                Tensor filled(const std::vector<int64_t>& dims, const std::vector<T>& values)
                {
                    Tensor t = engine->create_tensor(
                        SushiRuntime::span<const int64_t>(dims.data(), dims.size()),
                        sushiblas_dtype<T>());
                    TestEnvironment::queue()
                        .memcpy(t.data(), values.data(), values.size() * sizeof(T))
                        .wait();
                    return t;
                }

                /** @brief Creates a tensor of the extents of t, holding nothing yet. */
                Tensor like(const Tensor& t)
                {
                    return engine->create_tensor(
                        SushiRuntime::span<const int64_t>(t.shape.data(),
                                                          static_cast<std::size_t>(t.rank)),
                        t.dtype);
                }

                /** @brief Records the operation through the view. */
                TaskHandle record(Arithmetic which, const Tensor& a, const Tensor& v, Tensor& c,
                                  BroadcastAxis axis)
                {
                    BroadcastOps ops(recorder());
                    switch (which)
                    {
                        case Arithmetic::Add:
                            return ops.add(a, v, c, axis);
                        case Arithmetic::Sub:
                            return ops.sub(a, v, c, axis);
                        case Arithmetic::Mul:
                            return ops.mul(a, v, c, axis);
                        case Arithmetic::Div:
                            return ops.div(a, v, c, axis);
                    }
                    return TaskHandle();
                }

                /** @brief Returns the extents of the vector that goes with the row on the axis. */
                static const std::vector<int64_t>& vector_dims(const ShapeRow& row,
                                                               BroadcastAxis axis)
                {
                    return axis == BroadcastAxis::Leading ? row.leading : row.trailing;
                }

                /** @brief Expects the operation to equal a host loop at every shape row, in both modes. */
                void expect_matches_host_loop(Arithmetic which, BroadcastAxis axis)
                {
                    for (const ShapeRow& row : shape_rows())
                    {
                        const std::vector<T> a_values =
                            quarters<T>(element_count(row.dims), 4);
                        const std::vector<T> v_values =
                            vector_values<T>(which, element_count(vector_dims(row, axis)));
                        const std::vector<double> expected =
                            on_the_host(which, axis, a_values, v_values);

                        for (const ExecutionMode mode : K_FORCED_MODES)
                        {
                            const Tensor a = filled(row.dims, a_values);
                            const Tensor v = filled(vector_dims(row, axis), v_values);
                            Tensor c = like(a);

                            set_mode(mode);
                            EXPECT_TRUE(record(which, a, v, c, axis).is_valid());
                            recording->run().value();

                            EXPECT_EQ(widened(read_tensor<T>(c)), expected)
                                << "rank " << row.dims.size() << ", mode "
                                << (mode == ExecutionMode::ForceReference ? "reference" : "device");
                        }
                    }
                }

                /** @brief Expects the operation to equal a host loop when c is a, in both modes. */
                void expect_in_place_matches_host_loop(Arithmetic which, BroadcastAxis axis,
                                                       OutputAs output)
                {
                    for (const ShapeRow& row : shape_rows())
                    {
                        const std::vector<T> a_values =
                            quarters<T>(element_count(row.dims), 4);
                        const std::vector<T> v_values =
                            vector_values<T>(which, element_count(vector_dims(row, axis)));
                        const std::vector<double> expected =
                            on_the_host(which, axis, a_values, v_values);

                        for (const ExecutionMode mode : K_FORCED_MODES)
                        {
                            Tensor a = filled(row.dims, a_values);
                            const Tensor v = filled(vector_dims(row, axis), v_values);
                            Tensor view_of_a = a;
                            Tensor& c = output == OutputAs::TheSameObject ? a : view_of_a;

                            set_mode(mode);
                            EXPECT_TRUE(record(which, a, v, c, axis).is_valid());
                            recording->run().value();

                            EXPECT_EQ(widened(read_tensor<T>(a)), expected)
                                << "rank " << row.dims.size();
                        }
                    }
                }
        };

        using BroadcastOpsTypes = ::testing::Types<sycl::half, float, double>;
        SB_TYPED_TEST_SUITE(Medium_BroadcastOpsTyped, BroadcastOpsTypes);

        TYPED_TEST(Medium_BroadcastOpsTyped, AddOfALeadingVectorMatchesAHostLoopAtRanks2To4)
        {
            this->expect_matches_host_loop(Arithmetic::Add, BroadcastAxis::Leading);
        }

        TYPED_TEST(Medium_BroadcastOpsTyped, AddOfATrailingVectorMatchesAHostLoopAtRanks2To4)
        {
            this->expect_matches_host_loop(Arithmetic::Add, BroadcastAxis::Trailing);
        }

        TYPED_TEST(Medium_BroadcastOpsTyped, SubOfALeadingVectorMatchesAHostLoopAtRanks2To4)
        {
            this->expect_matches_host_loop(Arithmetic::Sub, BroadcastAxis::Leading);
        }

        TYPED_TEST(Medium_BroadcastOpsTyped, SubOfATrailingVectorMatchesAHostLoopAtRanks2To4)
        {
            this->expect_matches_host_loop(Arithmetic::Sub, BroadcastAxis::Trailing);
        }

        TYPED_TEST(Medium_BroadcastOpsTyped, MulOfALeadingVectorMatchesAHostLoopAtRanks2To4)
        {
            this->expect_matches_host_loop(Arithmetic::Mul, BroadcastAxis::Leading);
        }

        TYPED_TEST(Medium_BroadcastOpsTyped, MulOfATrailingVectorMatchesAHostLoopAtRanks2To4)
        {
            this->expect_matches_host_loop(Arithmetic::Mul, BroadcastAxis::Trailing);
        }

        TYPED_TEST(Medium_BroadcastOpsTyped, DivByALeadingVectorMatchesAHostLoopAtRanks2To4)
        {
            this->expect_matches_host_loop(Arithmetic::Div, BroadcastAxis::Leading);
        }

        TYPED_TEST(Medium_BroadcastOpsTyped, DivByATrailingVectorMatchesAHostLoopAtRanks2To4)
        {
            this->expect_matches_host_loop(Arithmetic::Div, BroadcastAxis::Trailing);
        }

        TYPED_TEST(Medium_BroadcastOpsTyped, COnTheSameObjectAsAGivesTheResultOfADistinctC)
        {
            this->expect_in_place_matches_host_loop(Arithmetic::Sub, BroadcastAxis::Leading,
                                                    OutputAs::TheSameObject);
        }

        TYPED_TEST(Medium_BroadcastOpsTyped, COnAnotherViewOfAsMemoryGivesTheResultOfADistinctC)
        {
            this->expect_in_place_matches_host_loop(Arithmetic::Div, BroadcastAxis::Trailing,
                                                    OutputAs::AnotherView);
        }
    } // namespace Test
} // namespace SushiBLAS
