/**************************************************************************/
/* test_rotg.cpp                                                          */
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
 * @file test_rotg.cpp
 * @brief Tests rotg against hand-computed rotations, across both kernels and inside a graph.
 * @author Mustafa Garip
 */

#include <cmath>
#include <random>
#include <string>
#include <vector>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <utility>
#include <algorithm>
#include <type_traits>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/core/traits.hpp>
#include <SushiBLAS/arithmetic/accumulator.hpp>
#include <SushiBLAS/arithmetic/complex_arithmetic.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_comparison.hpp>
#include <SushiBLAS/tensor_expectations/tensor_readback.hpp>
#include <SushiBLAS/tensor_expectations/scalar_dtype.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
#include "support/tolerance.hpp"
#include "blas_rows.hpp"
#include "rotg_table.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        using namespace RotgTable;

        namespace
        {
            /// @brief Seed of every random draw in this file.
            constexpr std::uint32_t K_SEED = 20261007u;

            /// @brief Random pairs the two kernels are compared on, per element type.
            constexpr int K_RANDOM_PAIRS = 100;

            /// @brief Largest magnitude of a drawn component.
            constexpr double K_DRAW_RANGE = 1.0;

            /// @brief Preset of every tensor a call must overwrite.
            constexpr double K_POISON = 999.0;

            /// @brief Rounded operations two kernels may differ by: 4 ulp, which is 8 roundoffs.
            constexpr double K_AGREEMENT_ROUNDINGS = 8.0;

            /// @brief Ratio above which two kernels disagree; the bound holds the slack.
            constexpr double K_AGREEMENT_THRESHOLD = 1.0;

            /// @brief Roundoffs, at the magnitude of a row, a complex rotation may be off it by.
            constexpr double K_TABLE_ROUNDINGS = 1.0;

            /// @brief Ratio above which a complex rotation misses its row: 8 epsilons.
            constexpr double K_TABLE_THRESHOLD = 16.0;

            /// @brief Length of the vectors whose dot product feeds a rotation.
            constexpr int64_t K_DOT_LENGTH = 3;

            /// @brief The dot product of K_DOT_LENGTH ones with themselves.
            constexpr double K_DOT_VALUE = 3.0;

            /// @brief Second operand of the rotation fed by the dot product.
            constexpr double K_DOT_PARTNER = 4.0;

            /// @brief First operand of the HALF pair whose squares overflow HALF.
            constexpr double K_LARGE_A = 300.0;

            /// @brief Second operand of the HALF pair whose squares overflow HALF.
            constexpr double K_LARGE_B = 400.0;

            /// @brief The element types whose rotation overwrites b with z.
            const std::vector<Core::DataType> K_REAL_TYPES = {
                Core::DataType::Half, Core::DataType::Float32, Core::DataType::Float64};

            /// @brief The kernel the cases that compare the device with the reference run.
            const std::vector<ExecutionMode> K_DEVICE_MODE = {ExecutionMode::ForceSYCL};

            /// @brief The kernel of the HALF case.
            const std::vector<ExecutionMode> K_REFERENCE_MODE = {ExecutionMode::ForceReference};

            /// @brief Both kernels an operation has.
            const std::vector<ExecutionMode> K_BOTH_MODES = {ExecutionMode::ForceReference,
                                                             ExecutionMode::ForceSYCL};

            /** @brief Holds one row of the complex table of the design, section 10.2. */
            struct ComplexRotationRow
            {
                std::complex<double> f; /**< The first operand. */
                std::complex<double> g; /**< The second operand. */
                std::complex<double> r; /**< The expected r. */
                double c;               /**< The expected cosine. */
                std::complex<double> s; /**< The expected sine. */
            };

            /// @brief The complex table of the design, section 10.2.
            const ComplexRotationRow K_COMPLEX_ROWS[] = {
                {{3.0, 0.0}, {4.0, 0.0}, {5.0, 0.0}, 0.6, {0.8, 0.0}},
                {{0.0, 3.0}, {4.0, 0.0}, {0.0, 5.0}, 0.6, {0.0, 0.8}},
                {{0.0, 0.0}, {3.0, 4.0}, {5.0, 0.0}, 0.0, {0.6, -0.8}},
                {{0.0, 0.0}, {0.0, 4.0}, {4.0, 0.0}, 0.0, {0.0, -1.0}},
                {{1.0, 2.0}, {0.0, 0.0}, {1.0, 2.0}, 1.0, {0.0, 0.0}}};

            /** @brief Names a row of a table by its place, as the axis of a suite. */
            struct TableRow
            {
                std::size_t index; /**< The place of the row in its table. */
            };

            /** @brief Returns the tail of a row named by its place in a table. */
            [[nodiscard]] std::string axis_name(const TableRow& row)
            {
                return "_row" + std::to_string(row.index);
            }

            /** @brief Returns the places of the first count rows of a table. */
            [[nodiscard]] std::vector<TableRow> table_rows(std::size_t count)
            {
                std::vector<TableRow> rows;
                for (std::size_t index = 0; index < count; ++index)
                    rows.push_back({index});
                return rows;
            }

            /** @brief Returns real + imaginary * i in T; a real T drops the imaginary part. */
            template<typename T>
            [[nodiscard]] T make(double real, [[maybe_unused]] double imaginary = 0.0)
            {
                if constexpr (Core::is_complex_v<T>)
                {
                    using R = typename T::value_type;
                    return T(static_cast<R>(real), static_cast<R>(imaginary));
                }
                else if constexpr (std::is_same_v<T, sycl::half>)
                    return sycl::half(static_cast<float>(real));
                else
                    return static_cast<T>(real);
            }

            /** @brief Returns a value whose every part is drawn from the draw range. */
            template<typename T>
            [[nodiscard]] T draw(std::mt19937& generator)
            {
                std::uniform_real_distribution<double> distribution(-K_DRAW_RANGE, K_DRAW_RANGE);
                const double real = distribution(generator);
                const double imaginary = distribution(generator);
                return make<T>(real, imaginary);
            }

            /** @brief Returns count pairs of drawn values. */
            template<typename T>
            [[nodiscard]] std::vector<std::pair<T, T>> draw_pairs(int count)
            {
                std::mt19937 generator(K_SEED);
                std::vector<std::pair<T, T>> pairs;
                pairs.reserve(static_cast<std::size_t>(count));
                for (int i = 0; i < count; ++i)
                {
                    const T first = draw<T>(generator);
                    const T second = draw<T>(generator);
                    pairs.emplace_back(first, second);
                }
                return pairs;
            }

            /** @brief Holds the four scalar tensors of one call as read after the graph ran. */
            template<typename T>
            struct Rotation
            {
                T a;                    /**< What a holds: r. */
                T b;                    /**< What b holds: z for a real T, b itself for a complex one. */
                Kernels::RealType<T> c; /**< The cosine. */
                T s;                    /**< The sine. */
            };

            /** @brief Returns a part of a value widened to double: 0 the real, 1 the imaginary. */
            template<typename V>
            [[nodiscard]] double part_of(const V& value, int part)
            {
                if constexpr (Core::is_complex_v<V>)
                {
                    const std::complex<double> wide = Detail::to_complex_double(value);
                    return part == 0 ? wide.real() : wide.imag();
                }
                else
                    return part == 0 ? Detail::to_double(value) : 0.0;
            }

            /**
             * @brief Expects every part of actual within the rounding bound two kernels of the
             *        component type of V may differ by.
             *
             * The bound is K_AGREEMENT_ROUNDINGS * unit roundoff * the larger modulus of the two
             * values, floored at the least normal of the component type, measured through
             * TestSupport::measure_error.
             */
            template<typename V>
            void expect_agreement(const V& actual, const V& expected, const char* what)
            {
                using R = Kernels::RealType<V>;
                const double least_normal = std::ldexp(
                    1.0, std::is_same_v<R, sycl::half> ? -14 : (std::is_same_v<R, float> ? -126 : -1022));
                for (int part = 0; part < 2; ++part)
                {
                    const double magnitude =
                        std::max({std::fabs(part_of(actual, part)), std::fabs(part_of(expected, part)),
                                  least_normal});
                    const double bound = K_AGREEMENT_ROUNDINGS * unit_roundoff_of<R>() * magnitude;
                    EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_error(
                        part_of(actual, part), part_of(expected, part), bound,
                        K_AGREEMENT_THRESHOLD)))
                        << what << ", part " << part;
                }
            }

            /**
             * @brief Expects a value within the rounding bound of a table row.
             *
             * The bound is K_TABLE_ROUNDINGS * unit roundoff * the larger of 1 and the modulus of
             * the row, held to K_TABLE_THRESHOLD.
             */
            template<typename R>
            void expect_table(const std::complex<R>& actual, const std::complex<double>& row,
                              const char* what)
            {
                const double bound =
                    K_TABLE_ROUNDINGS * unit_roundoff_of<R>() * std::max(1.0, std::abs(row));
                EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_error(
                    static_cast<double>(actual.real()), row.real(), bound, K_TABLE_THRESHOLD)))
                    << what << ", real part";
                EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_error(
                    static_cast<double>(actual.imag()), row.imag(), bound, K_TABLE_THRESHOLD)))
                    << what << ", imaginary part";
            }
        } // namespace

        /** @brief Records rotations on scalar tensors of element type T and reads them back. */
        class RotgStart : public BlasEngineStart
        {
            protected:
                /** @brief Returns a tensor of one element that holds value. */
                template<typename V>
                [[nodiscard]] Tensor scalar_tensor(const V& value)
                {
                    Tensor tensor = this->engine->create_tensor({1}, sushiblas_dtype<V>());
                    TestEnvironment::queue()
                        .memcpy(tensor.data(), &value, sizeof(V)).wait();
                    return tensor;
                }

                /** @brief Returns the rotation of each pair, all recorded in one graph. */
                template<typename T>
                [[nodiscard]] std::vector<Rotation<T>>
                rotate(const std::vector<std::pair<T, T>>& pairs)
                {
                    using Real = Kernels::RealType<T>;

                    std::vector<Tensor> a;
                    std::vector<Tensor> b;
                    std::vector<Tensor> c;
                    std::vector<Tensor> s;
                    a.reserve(pairs.size());
                    b.reserve(pairs.size());
                    c.reserve(pairs.size());
                    s.reserve(pairs.size());

                    SushiBLAS::BLASOps ops(this->recorder());
                    for (const std::pair<T, T>& pair : pairs)
                    {
                        a.push_back(scalar_tensor(pair.first));
                        b.push_back(scalar_tensor(pair.second));
                        c.push_back(scalar_tensor(make<Real>(K_POISON)));
                        s.push_back(scalar_tensor(make<T>(K_POISON, K_POISON)));
                        ops.rotg(a.back(), b.back(), c.back(), s.back());
                    }
                    this->recording->run().value();

                    std::vector<Rotation<T>> rotations;
                    rotations.reserve(pairs.size());
                    for (std::size_t i = 0; i < pairs.size(); ++i)
                        rotations.push_back(Rotation<T>{read_tensor<T>(a[i])[0],
                                                        read_tensor<T>(b[i])[0],
                                                        read_tensor<Real>(c[i])[0],
                                                        read_tensor<T>(s[i])[0]});
                    return rotations;
                }
        };

        /** @brief Runs one row of the real table per layout, kernel and real element type. */
        class Medium_RotgRealTable : public BlasRowCase<RotgStart, TableRow>
        {

        };

        /** @brief Runs the graph-ordering case per layout, kernel and real element type. */
        class Medium_RotgRealGraph : public BlasRowCase<RotgStart, NoAxis>
        {

        };

        /** @brief Runs one row of the complex table per layout, kernel and complex element type. */
        class Medium_RotgComplexTable : public BlasRowCase<RotgStart, TableRow>
        {

        };

        /** @brief Runs the kernel comparison per layout and element type. */
        class Medium_RotgModes : public BlasRowCase<RotgStart, NoAxis>
        {

        };

        /** @brief Runs the HALF rotation per layout on the reference kernel. */
        class Medium_RotgHalf : public BlasRowCase<RotgStart, NoAxis>
        {

        };

        /** @brief Tests the real rotation against a row of the design table; a and b receive r and z. */
        TEST_P(Medium_RotgRealTable, TheRotationMatchesTheTableRow)
        {
            const RealRotationRow& row = K_REAL_ROWS[GetParam().axis.index];
            with_real_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;

                const auto rotations = rotate<T>({{make<T>(row.a), make<T>(row.b)}});

                const Expected expected = expected_of<T>(row);
                EXPECT_EQ(Detail::to_double(rotations[0].a), expected.r);
                EXPECT_EQ(Detail::to_double(rotations[0].b), expected.z);
                EXPECT_EQ(Detail::to_double(rotations[0].c), expected.c);
                EXPECT_EQ(Detail::to_double(rotations[0].s), expected.s);
            });
        }

        INSTANTIATE_TEST_SUITE_P(
            , Medium_RotgRealTable,
            ::testing::ValuesIn(rows_of(K_REAL_TYPES, K_BOTH_MODES,
                                        table_rows(std::size(K_REAL_ROWS)))),
            row_name<TableRow>);

        /** @brief Tests that rotg reads an operand an earlier task of the same graph writes. */
        TEST_P(Medium_RotgRealGraph, AnInputWrittenByAnEarlierTaskIsRead)
        {
            const RealRotationRow row = {K_DOT_VALUE, K_DOT_PARTNER, 5.0, 3.0, 5.0, 4.0, 5.0,
                                         RebuildValue::InverseCosine};
            with_real_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;

                const Expected expected = expected_of<T>(row);

                Tensor ones = this->engine->create_tensor({K_DOT_LENGTH}, sushiblas_dtype<T>());
                const std::vector<T> one_values(static_cast<std::size_t>(K_DOT_LENGTH),
                                                make<T>(1.0));
                TestEnvironment::queue()
                    .memcpy(ones.data(), one_values.data(), one_values.size() * sizeof(T))
                    .wait();

                Tensor a = scalar_tensor(make<T>(K_POISON));
                Tensor b = scalar_tensor(make<T>(K_DOT_PARTNER));
                Tensor c = scalar_tensor(make<T>(K_POISON));
                Tensor s = scalar_tensor(make<T>(K_POISON));

                SushiBLAS::BLASOps ops(this->recorder());
                ops.dot(ones, ones, a);
                ops.rotg(a, b, c, s);
                this->recording->run().value();

                EXPECT_EQ(Detail::to_double(read_tensor<T>(a)[0]), expected.r);
                EXPECT_EQ(Detail::to_double(read_tensor<T>(b)[0]), expected.z);
                EXPECT_EQ(Detail::to_double(read_tensor<T>(c)[0]), expected.c);
                EXPECT_EQ(Detail::to_double(read_tensor<T>(s)[0]), expected.s);
            });
        }

        INSTANTIATE_TEST_SUITE_P(
            , Medium_RotgRealGraph, ::testing::ValuesIn(plain_rows(K_REAL_TYPES, K_BOTH_MODES)),
            row_name<NoAxis>);

        /** @brief Tests the complex rotation against a row of the design table; b is only read. */
        TEST_P(Medium_RotgComplexTable, TheRotationMatchesTheTableRow)
        {
            const ComplexRotationRow& row = K_COMPLEX_ROWS[GetParam().axis.index];
            with_complex_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;
                using R = typename T::value_type;

                const T f = make<T>(row.f.real(), row.f.imag());
                const T g = make<T>(row.g.real(), row.g.imag());
                const auto rotations = rotate<T>({{f, g}});

                expect_table<R>(rotations[0].a, row.r, "r");
                expect_table<R>(std::complex<R>(rotations[0].c, R(0)),
                                std::complex<double>(row.c, 0.0), "c");
                expect_table<R>(rotations[0].s, row.s, "s");
                EXPECT_EQ(0, std::memcmp(&rotations[0].b, &g, sizeof(T)));
            });
        }

        INSTANTIATE_TEST_SUITE_P(
            , Medium_RotgComplexTable,
            ::testing::ValuesIn(rows_of(K_COMPLEX_TYPES, K_BOTH_MODES,
                                        table_rows(std::size(K_COMPLEX_ROWS)))),
            row_name<TableRow>);

        /** @brief Tests that the two kernels give rotations within the agreement bound of each other. */
        TEST_P(Medium_RotgModes, BothModesAgree)
        {
            with_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;

                const std::vector<std::pair<T, T>> pairs = draw_pairs<T>(K_RANDOM_PAIRS);

                start(GetParam().layout, ExecutionMode::ForceReference);
                const auto reference = rotate<T>(pairs);
                start(GetParam().layout, ExecutionMode::ForceSYCL);
                const auto device = rotate<T>(pairs);

                ASSERT_EQ(device.size(), reference.size());
                for (std::size_t i = 0; i < device.size(); ++i)
                {
                    SCOPED_TRACE(::testing::Message() << "pair " << i);
                    expect_agreement(device[i].a, reference[i].a, "r");
                    expect_agreement(device[i].c, reference[i].c, "c");
                    expect_agreement(device[i].s, reference[i].s, "s");

                    if constexpr (Core::is_complex_v<T>)
                        EXPECT_EQ(0, std::memcmp(&device[i].b, &pairs[i].second, sizeof(T)));
                    else
                        expect_agreement(device[i].b, reference[i].b, "z");
                }
            });
        }

        INSTANTIATE_TEST_SUITE_P(
            , Medium_RotgModes, ::testing::ValuesIn(plain_rows(K_ALL_TYPES, K_DEVICE_MODE)),
            row_name<NoAxis>);

        /**
         * @brief Tests that the reference kernel forms a HALF rotation in FLOAT32: the squares of
         *        300 and 400 overflow HALF, yet r is 500 and each result is rounded to HALF once.
         */
        TEST_P(Medium_RotgHalf, AHalfRotationIsComputedInFloat)
        {
            // c = 0.6, s = 0.8 and z = 1 / c are formed in FLOAT32 and rounded to HALF once.
            constexpr float K_EXPECTED_R = 500.0f;
            constexpr float K_EXPECTED_C = 0.60009765625f;
            constexpr float K_EXPECTED_S = 0.7998046875f;
            constexpr float K_EXPECTED_Z = 1.6669921875f;

            const auto rotations = rotate<sycl::half>(
                {{make<sycl::half>(K_LARGE_A), make<sycl::half>(K_LARGE_B)}});

            EXPECT_EQ(static_cast<float>(rotations[0].a), K_EXPECTED_R);
            EXPECT_EQ(static_cast<float>(rotations[0].b), K_EXPECTED_Z);
            EXPECT_EQ(static_cast<float>(rotations[0].c), K_EXPECTED_C);
            EXPECT_EQ(static_cast<float>(rotations[0].s), K_EXPECTED_S);
        }

        INSTANTIATE_TEST_SUITE_P(
            , Medium_RotgHalf,
            ::testing::ValuesIn(plain_rows({Core::DataType::Half}, K_REFERENCE_MODE)),
            row_name<NoAxis>);
    } // namespace Test
} // namespace SushiBLAS
