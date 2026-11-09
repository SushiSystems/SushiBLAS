/**************************************************************************/
/* test_rotm.cpp                                                          */
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
 * @file test_rotm.cpp
 * @brief Tests rotm on each flag, on strided and empty vectors, across kernels and after rotmg.
 * @author Mustafa Garip
 */

#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <iterator>
#include <type_traits>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_comparison.hpp>
#include <SushiBLAS/tensor_expectations/tensor_readback.hpp>
#include <SushiBLAS/tensor_expectations/scalar_dtype.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
#include "blas_rows.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief Seed of every random draw in this file.
            constexpr std::uint32_t K_SEED = 20261007u;

            /// @brief Elements of a rotm parameter tensor: the flag and the four entries of H.
            constexpr int64_t K_PARAMETERS = 5;

            /// @brief Fills a parameter slot the flag implies, which a kernel must not read.
            constexpr double K_UNREAD = 99.0;

            /// @brief Fills the elements a strided view skips in x.
            constexpr double K_SKIPPED_IN_X = 77.0;

            /// @brief Fills the elements a strided view skips in y.
            constexpr double K_SKIPPED_IN_Y = 88.0;

            /// @brief Pairs the two kernels are compared on.
            constexpr int64_t K_LARGE_LENGTH = 100000;

            /// @brief A full H, flag -1, whose entries HALF holds exactly.
            constexpr double K_FULL_PARAM[K_PARAMETERS] = {-1.0, 0.5, -0.25, 0.75, 1.25};

            /** @brief Holds one parameter array and what it makes of x = [1, 2], y = [3, 4]. */
            struct RotmRow
            {
                double param[K_PARAMETERS]; /**< The flag, then h11, h21, h12 and h22. */
                double x[2];                /**< x after the call. */
                double y[2];                /**< y after the call. */
            };

            /// @brief The table of the design, section 10.3.
            constexpr RotmRow K_ROWS[] = {
                {{-1.0, 2.0, 4.0, 3.0, 5.0}, {11.0, 16.0}, {19.0, 28.0}},
                {{0.0, K_UNREAD, 4.0, 3.0, K_UNREAD}, {10.0, 14.0}, {7.0, 12.0}},
                {{1.0, 2.0, K_UNREAD, K_UNREAD, 5.0}, {5.0, 8.0}, {14.0, 18.0}},
                {{-2.0, K_UNREAD, K_UNREAD, K_UNREAD, K_UNREAD}, {1.0, 2.0}, {3.0, 4.0}}};

            /// @brief The suffixes of the rows of K_ROWS, in order: the flag each row carries.
            constexpr const char* K_FLAG_WORDS[] = {"FlagMinusOne", "FlagZero", "FlagOne",
                                                    "FlagMinusTwo"};

            /** @brief Returns value in the real element type T. */
            template<typename T>
            [[nodiscard]] T make(double value)
            {
                if constexpr (std::is_same_v<T, sycl::half>)
                    return sycl::half(static_cast<float>(value));
                else
                    return static_cast<T>(value);
            }

            /** @brief Returns values in the real element type T. */
            template<typename T, std::size_t N>
            [[nodiscard]] std::vector<T> make_all(const double (&values)[N])
            {
                std::vector<T> out;
                out.reserve(N);
                for (const double value : values)
                    out.push_back(make<T>(value));
                return out;
            }

            /** @brief Expects every element to equal its expectation with no tolerance. */
            template<typename T>
            void expect_equal(const std::vector<T>& actual, const std::vector<T>& expected)
            {
                ASSERT_EQ(actual.size(), expected.size());
                for (std::size_t i = 0; i < actual.size(); ++i)
                    EXPECT_EQ(Detail::to_double(actual[i]), Detail::to_double(expected[i]))
                        << "index " << i;
            }

            /** @brief Holds how many elements of two vectors differ and the first index that does. */
            struct Differences
            {
                std::size_t count; /**< The number of elements that differ. */
                std::size_t first; /**< The first index that differs, 0 when none does. */
            };

            /** @brief Returns the elements at which two vectors of one length differ. */
            template<typename T>
            [[nodiscard]] Differences differences(const std::vector<T>& a, const std::vector<T>& b)
            {
                Differences found{0, 0};
                for (std::size_t i = 0; i < a.size(); ++i)
                {
                    if (Detail::to_double(a[i]) == Detail::to_double(b[i]))
                        continue;
                    if (found.count == 0)
                        found.first = i;
                    ++found.count;
                }
                return found;
            }
        } // namespace

        /// @brief Holds the index of the row of K_ROWS that completes a row of the flag suite.
struct FlagAxis
{
    std::size_t index; /**< The row of K_ROWS the case applies. */
};

/** @brief Returns the row of K_ROWS that @p axis names. */
[[nodiscard]] inline const RotmRow& table_row(const FlagAxis& axis)
{
    return K_ROWS[axis.index];
}

        /** @brief Returns the tail of a row named by the flag of its parameter array. */
        [[nodiscard]] inline std::string axis_name(const FlagAxis& axis)
        {
            return std::string("_") + K_FLAG_WORDS[axis.index];
        }

        /**
         * @brief Builds the vectors and the parameter tensor of a rotm call.
         */
        class RotmStart : public BlasEngineStart
        {
            protected:
                /** @brief Returns a rank-1 tensor that holds values. */
                template<typename T>
                [[nodiscard]] Tensor vector_of(const std::vector<T>& values)
                {
                    Tensor tensor = engine->create_tensor(
                        {static_cast<int64_t>(values.size())}, sushiblas_dtype<T>());
                    if (!values.empty())
                        TestEnvironment::queue()
                            .memcpy(tensor.data(), values.data(), values.size() * sizeof(T))
                            .wait();
                    return tensor;
                }

                /** @brief Returns a view of every second element of a rank-1 parent. */
                [[nodiscard]] static Tensor every_second_element(const Tensor& parent)
                {
                    const int64_t dims[] = {parent.num_elements / 2};
                    std::array<int64_t, Core::K_MAX_TENSOR_RANK> strides{};
                    strides[0] = 2;
                    return Tensor(parent.storage, SushiRuntime::span<const int64_t>(dims, 1),
                                  strides, parent.dtype, 0, parent.layout);
                }

                /** @brief Returns count values drawn from [-1, 1]. */
                template<typename T>
                [[nodiscard]] static std::vector<T> random_values(std::mt19937& generator,
                                                                  int64_t count)
                {
                    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
                    std::vector<T> values(static_cast<std::size_t>(count));
                    for (T& value : values)
                        value = make<T>(distribution(generator));
                    return values;
                }

                /** @brief Rotates x and y by the full H in the layout and kernel given; x then y. */
                template<typename T>
                [[nodiscard]] std::pair<std::vector<T>, std::vector<T>>
                rotated_by_the_full_h(Core::Layout layout, ExecutionMode mode,
                                      const std::vector<T>& x_values,
                                      const std::vector<T>& y_values)
                {
                    start(layout, mode);
                    Tensor x = vector_of(x_values);
                    Tensor y = vector_of(y_values);
                    const Tensor param = vector_of(make_all<T>(K_FULL_PARAM));

                    SushiBLAS::BLASOps(recorder()).rotm(x, y, param);
                    recording->run().value();

                    return {read_tensor<T>(x), read_tensor<T>(y)};
                }
        };

        /** @brief Runs rotm once per flag, over the real element types, layouts and kernels. */
        class Medium_RotmFlags : public BlasRowCase<RotmStart, FlagAxis>
        {

        };

        /** @brief Runs the rotm cases that need no flag axis, over the real element types. */
        class Medium_Rotm : public BlasRowCase<RotmStart, NoAxis>
        {

        };

        /** @brief Runs the comparison of the two kernels, over the real element types. */
        class Medium_RotmKernels : public BlasRowCase<RotmStart, NoAxis>
        {

        };

        /** @brief Runs rotm after rotmg over the element types rotmg accepts. */
        class Medium_RotmChained : public BlasRowCase<RotmStart, NoAxis>
        {

        };

        /** @brief Returns the rows of the flag suite: every flag of K_ROWS on every real type. */
        [[nodiscard]] inline std::vector<Row<FlagAxis>> flag_rows()
        {
            std::vector<FlagAxis> axes;
            for (std::size_t index = 0; index < std::size(K_ROWS); ++index)
                axes.push_back({index});
            return rows_of({Core::DataType::Half, Core::DataType::Float32, Core::DataType::Float64},
                           {ExecutionMode::ForceReference, ExecutionMode::ForceSYCL}, axes);
        }

        /** @brief Returns the rows over the real element types and both kernels. */
        [[nodiscard]] inline std::vector<Row<NoAxis>> real_rows()
        {
            return plain_rows({Core::DataType::Half, Core::DataType::Float32, Core::DataType::Float64},
                              {ExecutionMode::ForceReference, ExecutionMode::ForceSYCL});
        }

        /**
         * @brief Tests each flag against its hand-computed result; an implied slot holds
         *        K_UNREAD.
         */
        TEST_P(Medium_RotmFlags, EachFlagAppliesOnlyTheEntriesItDoesNotImply)
        {
            const RotmRow& row = table_row(GetParam().axis);
            with_real_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;

                Tensor x = vector_of<T>({make<T>(1.0), make<T>(2.0)});
                Tensor y = vector_of<T>({make<T>(3.0), make<T>(4.0)});
                const Tensor param = vector_of(make_all<T>(row.param));

                SushiBLAS::BLASOps(recorder()).rotm(x, y, param);
                recording->run().value();

                expect_equal(read_tensor<T>(x), make_all<T>(row.x));
                expect_equal(read_tensor<T>(y), make_all<T>(row.y));
            });
        }

        INSTANTIATE_TEST_SUITE_P(, Medium_RotmFlags, ::testing::ValuesIn(flag_rows()),
                                 row_name<FlagAxis>);

        /**
         * @brief Tests that flag -2 writes nothing: an infinity stays and makes no NaN.
         */
        TEST_P(Medium_Rotm, TheIdentityFlagLeavesAnInfinityAlone)
        {
            with_real_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;

                Tensor x = vector_of<T>(
                    {make<T>(std::numeric_limits<double>::infinity()), make<T>(2.0)});
                Tensor y = vector_of<T>({make<T>(3.0), make<T>(4.0)});
                const Tensor param = vector_of(make_all<T>(K_ROWS[3].param));

                SushiBLAS::BLASOps(recorder()).rotm(x, y, param);
                recording->run().value();

                const std::vector<T> x_after = read_tensor<T>(x);
                const std::vector<T> y_after = read_tensor<T>(y);
                EXPECT_TRUE(std::isinf(Detail::to_double(x_after[0])));
                EXPECT_GT(Detail::to_double(x_after[0]), 0.0);
                EXPECT_EQ(Detail::to_double(x_after[1]), 2.0);
                EXPECT_EQ(Detail::to_double(y_after[0]), 3.0);
                EXPECT_EQ(Detail::to_double(y_after[1]), 4.0);
            });
        }

        /**
         * @brief Tests that a view of every second element is transformed and the elements
         *        between are left as they were.
         */
        TEST_P(Medium_Rotm, StridedVectorsLeaveTheSkippedElementsAlone)
        {
            const double x_values[] = {1.0, K_SKIPPED_IN_X, 2.0, K_SKIPPED_IN_X};
            const double y_values[] = {3.0, K_SKIPPED_IN_Y, 4.0, K_SKIPPED_IN_Y};
            const double x_expected[] = {11.0, K_SKIPPED_IN_X, 16.0, K_SKIPPED_IN_X};
            const double y_expected[] = {19.0, K_SKIPPED_IN_Y, 28.0, K_SKIPPED_IN_Y};

            with_real_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;

                Tensor x_parent = vector_of(make_all<T>(x_values));
                Tensor y_parent = vector_of(make_all<T>(y_values));
                Tensor x = every_second_element(x_parent);
                Tensor y = every_second_element(y_parent);
                const Tensor param = vector_of(make_all<T>(K_ROWS[0].param));

                SushiBLAS::BLASOps(recorder()).rotm(x, y, param);
                recording->run().value();

                expect_equal(read_tensor<T>(x_parent), make_all<T>(x_expected));
                expect_equal(read_tensor<T>(y_parent), make_all<T>(y_expected));
            });
        }

        /**
         * @brief Tests that vectors of no elements record and run, and leave the parameter.
         */
        TEST_P(Medium_Rotm, AnEmptyPairRecordsAndRuns)
        {
            with_real_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;

                Tensor x = vector_of(std::vector<T>{});
                Tensor y = vector_of(std::vector<T>{});
                const Tensor param = vector_of(make_all<T>(K_FULL_PARAM));

                SushiBLAS::BLASOps(recorder()).rotm(x, y, param);

                EXPECT_TRUE(recording->run().has_value());
                expect_equal(read_tensor<T>(param), make_all<T>(K_FULL_PARAM));
            });
        }

        INSTANTIATE_TEST_SUITE_P(, Medium_Rotm, ::testing::ValuesIn(real_rows()),
                                 row_name<NoAxis>);

        /**
         * @brief Tests that the two kernels give the same elements on K_LARGE_LENGTH pairs
         *        under a full H.
         *
         * The row names the layout and the element type; its kernel is the device one, and the
         * reference run is made inside the case.
         */
        TEST_P(Medium_RotmKernels, BothModesAgreeElementForElement)
        {
            with_real_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;

                std::mt19937 generator(K_SEED);
                const std::vector<T> x_values = random_values<T>(generator, K_LARGE_LENGTH);
                const std::vector<T> y_values = random_values<T>(generator, K_LARGE_LENGTH);

                const auto reference = rotated_by_the_full_h<T>(
                    GetParam().layout, ExecutionMode::ForceReference, x_values, y_values);
                const auto device = rotated_by_the_full_h<T>(
                    GetParam().layout, ExecutionMode::ForceSYCL, x_values, y_values);

                const Differences in_x = differences(device.first, reference.first);
                const Differences in_y = differences(device.second, reference.second);
                EXPECT_EQ(in_x.count, 0u) << "x differs first at index " << in_x.first;
                EXPECT_EQ(in_y.count, 0u) << "y differs first at index " << in_y.first;
            });
        }

        INSTANTIATE_TEST_SUITE_P(
            , Medium_RotmKernels,
            ::testing::ValuesIn(plain_rows({Core::DataType::Half, Core::DataType::Float32,
                                            Core::DataType::Float64},
                                           {ExecutionMode::ForceSYCL})),
            row_name<NoAxis>);

        /**
         * @brief Tests that rotm reads the param an earlier rotmg of the same graph writes.
         */
        TEST_P(Medium_RotmChained, AParamWrittenByRotmgIsRead)
        {
            const double unwritten[K_PARAMETERS] = {K_UNREAD, K_UNREAD, K_UNREAD, K_UNREAD,
                                                    K_UNREAD};

            with_real_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;

                Tensor d1 = vector_of<T>({make<T>(5.0)});
                Tensor d2 = vector_of<T>({make<T>(5.0)});
                Tensor x1 = vector_of<T>({make<T>(4.0)});
                const Tensor y1 = vector_of<T>({make<T>(2.0)});
                Tensor param = vector_of(make_all<T>(unwritten));
                Tensor x = vector_of<T>({make<T>(4.0)});
                Tensor y = vector_of<T>({make<T>(2.0)});

                SushiBLAS::BLASOps ops(recorder());
                ops.rotmg(d1, d2, x1, y1, param);
                ops.rotm(x, y, param);
                recording->run().value();

                // H is [1, 0.5; -0.5, 1], so (4, 2) becomes (5, 0).
                expect_equal(read_tensor<T>(x), std::vector<T>{make<T>(5.0)});
                expect_equal(read_tensor<T>(y), std::vector<T>{make<T>(0.0)});
            });
        }

        INSTANTIATE_TEST_SUITE_P(
            , Medium_RotmChained,
            ::testing::ValuesIn(plain_rows({Core::DataType::Float32, Core::DataType::Float64},
                                           {ExecutionMode::ForceReference, ExecutionMode::ForceSYCL})),
            row_name<NoAxis>);
    } // namespace Test
} // namespace SushiBLAS
