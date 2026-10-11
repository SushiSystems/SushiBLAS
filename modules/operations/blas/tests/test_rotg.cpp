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
 * @file test_rotg.cpp
 * @brief Tests rotg against hand-computed rotations, across both kernels and inside a graph.
 * @author Mustafa Garip
 */

#include <cmath>
#include <limits>
#include <random>
#include <string>
#include <vector>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <algorithm>
#include <type_traits>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/SushiBLAS.h>
#include <SushiBLAS/core/traits.hpp>
#include <SushiBLAS/blas_seam_kernels/givens.hpp>
#include <SushiBLAS/support/accumulator.hpp>
#include <SushiBLAS/support/complex_arithmetic.hpp>
#include "test_utils.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief Seed of every random draw in this file.
            constexpr std::uint32_t K_SEED = 20261007u;

            /// @brief Random pairs the two kernels are compared on, per element type.
            constexpr int K_RANDOM_PAIRS = 100;

            /// @brief Random HALF pairs compared against the rotation evaluated in FLOAT32.
            constexpr int K_RANDOM_HALF_PAIRS = 50;

            /// @brief Largest magnitude of a drawn component.
            constexpr double K_DRAW_RANGE = 1.0;

            /// @brief Preset of every tensor a call must overwrite.
            constexpr double K_POISON = 999.0;

            /// @brief Units in the last place the two kernels may differ by: three a device square root may be off by, one for the quotient.
            constexpr double K_MODE_ULPS = 4.0;

            /// @brief Length of the vectors whose dot product feeds a rotation.
            constexpr int64_t K_DOT_LENGTH = 3;

            /// @brief The dot product of K_DOT_LENGTH ones with themselves.
            constexpr double K_DOT_VALUE = 3.0;

            /// @brief Second operand of the rotation fed by the dot product.
            constexpr double K_DOT_PARTNER = 4.0;

            /**
             * @brief Allowed distance of a complex rotation from its table row, for operands of
             *        unit size: 8 units of the epsilon of R.
             */
            template<typename R>
            constexpr R K_TABLE_TOLERANCE = R(8) * std::numeric_limits<R>::epsilon();

            /// @brief Both kernels an operation has.
            constexpr ExecutionMode K_MODES[] = {ExecutionMode::ForceReference,
                                                 ExecutionMode::ForceSYCL};

            /// @brief Both element orders an engine can create tensors in.
            constexpr Core::Layout K_LAYOUTS[] = {Core::Layout::RowMajor,
                                                  Core::Layout::ColumnMajor};

            /** @brief Takes the square root of a rotation on the host. */
            struct StdRoot
            {
                /** @brief Returns the square root of value. */
                template<typename R>
                [[nodiscard]] static R apply(R value) noexcept
                {
                    return std::sqrt(value);
                }
            };

            /**
             * @brief Holds the binary precision of a real element type.
             * @tparam R sycl::half, float or double.
             */
            template<typename R>
            struct Precision;

            /** @brief Holds the precision of HALF. */
            template<>
            struct Precision<sycl::half>
            {
                static constexpr int K_DIGITS = 11;        /**< Bits of the significand. */
                static constexpr int K_MIN_EXPONENT = -14; /**< Exponent of the least normal. */
            };

            /** @brief Holds the precision of FLOAT32. */
            template<>
            struct Precision<float>
            {
                static constexpr int K_DIGITS = 24;         /**< Bits of the significand. */
                static constexpr int K_MIN_EXPONENT = -126; /**< Exponent of the least normal. */
            };

            /** @brief Holds the precision of FLOAT64. */
            template<>
            struct Precision<double>
            {
                static constexpr int K_DIGITS = 53;          /**< Bits of the significand. */
                static constexpr int K_MIN_EXPONENT = -1022; /**< Exponent of the least normal. */
            };

            /** @brief Names how z of a real rotation is rebuilt from its other values. */
            enum class RebuildValue : std::uint8_t
            {
                Zero,          /**< z is 0. */
                One,           /**< z is 1. */
                Sine,          /**< z is s. */
                InverseCosine /**< z is 1 / c. */
            };

            /** @brief Holds one row of the real table of the design, section 10.2. */
            struct RealRotationRow
            {
                double a;             /**< The first operand. */
                double b;             /**< The second operand. */
                double r;             /**< The expected r, which is exact. */
                double c_numerator;   /**< The expected c is this over c_denominator. */
                double c_denominator; /**< The divisor of the expected c. */
                double s_numerator;   /**< The expected s is this over s_denominator. */
                double s_denominator; /**< The divisor of the expected s. */
                RebuildValue z;       /**< The expected z. */
            };

            /** @brief Holds one row of the complex table of the design, section 10.2. */
            struct ComplexRotationRow
            {
                std::complex<double> f; /**< The first operand. */
                std::complex<double> g; /**< The second operand. */
                std::complex<double> r; /**< The expected r. */
                double c;               /**< The expected cosine. */
                std::complex<double> s; /**< The expected sine. */
            };

            /// @brief The real table of the design, section 10.2.
            constexpr RealRotationRow K_REAL_ROWS[] = {
                {3.0, 4.0, 5.0, 3.0, 5.0, 4.0, 5.0, RebuildValue::InverseCosine},
                {4.0, 3.0, 5.0, 4.0, 5.0, 3.0, 5.0, RebuildValue::Sine},
                {-4.0, 3.0, -5.0, -4.0, -5.0, 3.0, -5.0, RebuildValue::Sine},
                {3.0, -4.0, -5.0, 3.0, -5.0, -4.0, -5.0, RebuildValue::InverseCosine},
                {0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, RebuildValue::Zero},
                {0.0, 2.0, 2.0, 0.0, 1.0, 1.0, 1.0, RebuildValue::One},
                {2.0, 0.0, 2.0, 1.0, 1.0, 0.0, 1.0, RebuildValue::Zero}};

            /// @brief The complex table of the design, section 10.2.
            const ComplexRotationRow K_COMPLEX_ROWS[] = {
                {{3.0, 0.0}, {4.0, 0.0}, {5.0, 0.0}, 0.6, {0.8, 0.0}},
                {{0.0, 3.0}, {4.0, 0.0}, {0.0, 5.0}, 0.6, {0.0, 0.8}},
                {{0.0, 0.0}, {3.0, 4.0}, {5.0, 0.0}, 0.0, {0.6, -0.8}},
                {{0.0, 0.0}, {0.0, 4.0}, {4.0, 0.0}, 0.0, {0.0, -1.0}},
                {{1.0, 2.0}, {0.0, 0.0}, {1.0, 2.0}, 1.0, {0.0, 0.0}}};

            /** @brief Returns the name of a mode and a layout for a failure message. */
            [[nodiscard]] std::string engine_name(Core::Layout layout, ExecutionMode mode)
            {
                return std::string(mode == ExecutionMode::ForceSYCL ? "ForceSYCL"
                                                                     : "ForceReference") +
                       (layout == Core::Layout::RowMajor ? ", ROW_MAJOR" : ", COLUMN_MAJOR");
            }

            /** @brief Returns whether the test device computes in the component type of T. */
            template<typename T>
            [[nodiscard]] bool device_runs()
            {
                using R = Kernels::RealType<T>;
                const sycl::device device =
                    TestEnvironment::get_context().devices().queue(0).get_device();
                if constexpr (std::is_same_v<R, sycl::half>)
                    return device.has(sycl::aspect::fp16);
                else if constexpr (std::is_same_v<R, double>)
                    return device.has(sycl::aspect::fp64);
                else
                    return true;
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

            /** @brief Returns the spacing of the values of R at a magnitude. */
            template<typename R>
            [[nodiscard]] double ulp_at(double magnitude)
            {
                const int exponent =
                    std::max(std::ilogb(magnitude), Precision<R>::K_MIN_EXPONENT);
                return std::ldexp(1.0, exponent - (Precision<R>::K_DIGITS - 1));
            }

            /**
             * @brief Expects every part of actual within K_MODE_ULPS units in the last place of
             *        expected, the unit taken at the larger magnitude of the two.
             */
            template<typename V>
            void expect_within_ulps(const V& actual, const V& expected, const char* what)
            {
                using Wide = std::complex<double>;
                Wide a;
                Wide e;
                if constexpr (Core::is_complex_v<V>)
                {
                    a = Detail::to_complex_double(actual);
                    e = Detail::to_complex_double(expected);
                }
                else
                {
                    a = Wide(Detail::to_double(actual), 0.0);
                    e = Wide(Detail::to_double(expected), 0.0);
                }

                const double unit =
                    ulp_at<Kernels::RealType<V>>(std::max(std::abs(a), std::abs(e)));
                EXPECT_LE(std::abs(a.real() - e.real()), K_MODE_ULPS * unit) << what;
                EXPECT_LE(std::abs(a.imag() - e.imag()), K_MODE_ULPS * unit) << what;
            }
        } // namespace

        /**
         * @brief Records rotations on scalar tensors of element type T and reads them back.
         */
        template<typename T>
        class RotgCases : public SushiBLASTest
        {
            protected:
                /// @brief The component type of T, which the cosine has.
                using Real = Kernels::RealType<T>;

                /** @brief Holds the four scalar tensors of one call as read after the graph ran. */
                struct Rotation
                {
                    T a;    /**< What a holds: r. */
                    T b;    /**< What b holds: z for a real T, b itself for a complex one. */
                    Real c; /**< The cosine. */
                    T s;    /**< The sine. */
                };

                /** @brief Skips the case on a device that lacks the component type of T. */
                void SetUp() override
                {
                    SushiBLASTest::SetUp();
                    if (!device_runs<T>())
                        GTEST_SKIP() << "the device does not compute in this element type";
                }

                /** @brief Rebuilds the engine in a layout and selects a kernel. */
                void start(Core::Layout layout, ExecutionMode mode)
                {
                    this->reinit_engine(layout);
                    this->engine->set_mode(mode);
                }

                /** @brief Returns a tensor of one element that holds value. */
                template<typename V>
                [[nodiscard]] Tensor scalar_tensor(const V& value)
                {
                    Tensor tensor = this->engine->create_tensor({1}, get_sushiblas_dtype<V>());
                    TestEnvironment::get_context().devices().queue(0)
                        .memcpy(tensor.data(), &value, sizeof(V)).wait();
                    return tensor;
                }

                /** @brief Returns the rotation of each pair, all recorded in one graph. */
                [[nodiscard]] std::vector<Rotation>
                rotate(const std::vector<std::pair<T, T>>& pairs)
                {
                    std::vector<Tensor> a;
                    std::vector<Tensor> b;
                    std::vector<Tensor> c;
                    std::vector<Tensor> s;
                    a.reserve(pairs.size());
                    b.reserve(pairs.size());
                    c.reserve(pairs.size());
                    s.reserve(pairs.size());

                    SushiBLAS::BLASOps ops(*this->engine);
                    for (const std::pair<T, T>& pair : pairs)
                    {
                        a.push_back(scalar_tensor(pair.first));
                        b.push_back(scalar_tensor(pair.second));
                        c.push_back(scalar_tensor(make<Real>(K_POISON)));
                        s.push_back(scalar_tensor(make<T>(K_POISON, K_POISON)));
                        ops.rotg(a.back(), b.back(), c.back(), s.back());
                    }
                    this->engine->execute();

                    std::vector<Rotation> rotations;
                    rotations.reserve(pairs.size());
                    for (std::size_t i = 0; i < pairs.size(); ++i)
                        rotations.push_back(Rotation{read_tensor<T>(a[i])[0],
                                                     read_tensor<T>(b[i])[0],
                                                     read_tensor<Real>(c[i])[0],
                                                     read_tensor<T>(s[i])[0]});
                    return rotations;
                }
        };

        /** @brief Runs rotg over the five element types. */
        template<typename T>
        class Unit_Rotg : public RotgCases<T>
        {

        };

        /** @brief Runs rotg over the real element types. */
        template<typename T>
        class Unit_RotgReal : public RotgCases<T>
        {
            protected:
                /// @brief The type a rotation of T is computed in.
                using Wide = Kernels::Accumulator<T>;

                /** @brief Holds what one row of the real table expects, rounded to T once. */
                struct Expected
                {
                    double r; /**< The expected r. */
                    double z; /**< The expected z. */
                    double c; /**< The expected cosine. */
                    double s; /**< The expected sine. */
                };

                /** @brief Returns a value of Wide as T holds it, widened to double. */
                [[nodiscard]] static double stored(Wide value)
                {
                    return Detail::to_double(static_cast<T>(value));
                }

                /** @brief Returns the expectation of a row, each quotient formed in Wide. */
                [[nodiscard]] static Expected expected_of(const RealRotationRow& row)
                {
                    const Wide c = static_cast<Wide>(row.c_numerator) /
                                   static_cast<Wide>(row.c_denominator);
                    const Wide s = static_cast<Wide>(row.s_numerator) /
                                   static_cast<Wide>(row.s_denominator);

                    Wide z = Wide(0);
                    switch (row.z)
                    {
                        case RebuildValue::Zero:
                            z = Wide(0);
                            break;
                        case RebuildValue::One:
                            z = Wide(1);
                            break;
                        case RebuildValue::Sine:
                            z = s;
                            break;
                        case RebuildValue::InverseCosine:
                            z = Wide(1) / c;
                            break;
                    }

                    return Expected{stored(static_cast<Wide>(row.r)), stored(z), stored(c),
                                    stored(s)};
                }
        };

        /** @brief Runs rotg over the complex element types. */
        template<typename T>
        class Unit_RotgComplex : public RotgCases<T>
        {

        };

        /// @brief Names the suite of the cases that run on HALF alone.
        using Unit_RotgHalf = RotgCases<sycl::half>;

        /// @brief Names the five element types rotg accepts.
        using RotgTypes = ::testing::Types<sycl::half, float, double, std::complex<float>,
                                           std::complex<double>>;
        /// @brief Names the real element types, whose rotation overwrites b with z.
        using RotgRealTypes = ::testing::Types<sycl::half, float, double>;
        /// @brief Names the complex element types, whose rotation leaves b as it is.
        using RotgComplexTypes = ::testing::Types<std::complex<float>, std::complex<double>>;
        SB_TYPED_TEST_SUITE(Unit_Rotg, RotgTypes);
        SB_TYPED_TEST_SUITE(Unit_RotgReal, RotgRealTypes);
        SB_TYPED_TEST_SUITE(Unit_RotgComplex, RotgComplexTypes);

        /**
         * @brief Tests the real rotation against the table of the design; a and b receive r
         *        and z.
         */
        TYPED_TEST(Unit_RotgReal, RealPairsMatchTheTable)
        {
            using T = TypeParam;

            std::vector<std::pair<T, T>> pairs;
            for (const RealRotationRow& row : K_REAL_ROWS)
                pairs.emplace_back(make<T>(row.a), make<T>(row.b));

            for (const Core::Layout layout : K_LAYOUTS)
            {
                for (const ExecutionMode mode : K_MODES)
                {
                    SCOPED_TRACE(engine_name(layout, mode));
                    this->start(layout, mode);
                    const auto rotations = this->rotate(pairs);

                    for (std::size_t i = 0; i < rotations.size(); ++i)
                    {
                        const RealRotationRow& row = K_REAL_ROWS[i];
                        SCOPED_TRACE(::testing::Message() << "a " << row.a << ", b " << row.b);

                        const auto expected = TestFixture::expected_of(row);
                        EXPECT_EQ(Detail::to_double(rotations[i].a), expected.r);
                        EXPECT_EQ(Detail::to_double(rotations[i].b), expected.z);
                        EXPECT_EQ(Detail::to_double(rotations[i].c), expected.c);
                        EXPECT_EQ(Detail::to_double(rotations[i].s), expected.s);
                    }
                }
            }
        }

        /**
         * @brief Tests that rotg reads an operand an earlier task of the same graph writes.
         */
        TYPED_TEST(Unit_RotgReal, AnInputWrittenByAnEarlierTaskIsRead)
        {
            using T = TypeParam;
            const RealRotationRow row = {K_DOT_VALUE, K_DOT_PARTNER, 5.0, 3.0, 5.0, 4.0, 5.0,
                                         RebuildValue::InverseCosine};
            const auto expected = TestFixture::expected_of(row);

            for (const Core::Layout layout : K_LAYOUTS)
            {
                for (const ExecutionMode mode : K_MODES)
                {
                    SCOPED_TRACE(engine_name(layout, mode));
                    this->start(layout, mode);

                    Tensor ones = this->engine->create_tensor({K_DOT_LENGTH},
                                                              get_sushiblas_dtype<T>());
                    const std::vector<T> one_values(static_cast<std::size_t>(K_DOT_LENGTH),
                                                    make<T>(1.0));
                    TestEnvironment::get_context().devices().queue(0)
                        .memcpy(ones.data(), one_values.data(), one_values.size() * sizeof(T))
                        .wait();

                    Tensor a = this->scalar_tensor(make<T>(K_POISON));
                    Tensor b = this->scalar_tensor(make<T>(K_DOT_PARTNER));
                    Tensor c = this->scalar_tensor(make<T>(K_POISON));
                    Tensor s = this->scalar_tensor(make<T>(K_POISON));

                    SushiBLAS::BLASOps ops(*this->engine);
                    ops.dot(ones, ones, a);
                    ops.rotg(a, b, c, s);
                    this->engine->execute();

                    EXPECT_EQ(Detail::to_double(read_tensor<T>(a)[0]), expected.r);
                    EXPECT_EQ(Detail::to_double(read_tensor<T>(b)[0]), expected.z);
                    EXPECT_EQ(Detail::to_double(read_tensor<T>(c)[0]), expected.c);
                    EXPECT_EQ(Detail::to_double(read_tensor<T>(s)[0]), expected.s);
                }
            }
        }

        /**
         * @brief Tests the complex rotation against the table of the design; b is only read.
         */
        TYPED_TEST(Unit_RotgComplex, ComplexPairsMatchTheTable)
        {
            using T = TypeParam;
            using R = typename T::value_type;

            std::vector<std::pair<T, T>> pairs;
            for (const ComplexRotationRow& row : K_COMPLEX_ROWS)
                pairs.emplace_back(make<T>(row.f.real(), row.f.imag()),
                                   make<T>(row.g.real(), row.g.imag()));

            for (const Core::Layout layout : K_LAYOUTS)
            {
                for (const ExecutionMode mode : K_MODES)
                {
                    SCOPED_TRACE(engine_name(layout, mode));
                    this->start(layout, mode);
                    const auto rotations = this->rotate(pairs);

                    for (std::size_t i = 0; i < rotations.size(); ++i)
                    {
                        const ComplexRotationRow& row = K_COMPLEX_ROWS[i];
                        SCOPED_TRACE(::testing::Message() << "f " << row.f << ", g " << row.g);

                        const T f = pairs[i].first;
                        const T g = pairs[i].second;
                        const R tolerance = K_TABLE_TOLERANCE<R> * (std::abs(f) + std::abs(g));
                        EXPECT_LE(std::abs(rotations[i].a - make<T>(row.r.real(), row.r.imag())),
                                  tolerance);
                        EXPECT_LE(std::abs(rotations[i].c - static_cast<R>(row.c)),
                                  K_TABLE_TOLERANCE<R>);
                        EXPECT_LE(std::abs(rotations[i].s - make<T>(row.s.real(), row.s.imag())),
                                  K_TABLE_TOLERANCE<R>);
                        EXPECT_EQ(0, std::memcmp(&rotations[i].b, &g, sizeof(T)));
                    }
                }
            }
        }

        /**
         * @brief Tests that the two kernels give rotations within K_MODE_ULPS of each other.
         */
        TYPED_TEST(Unit_Rotg, BothModesAgree)
        {
            using T = TypeParam;
            const std::vector<std::pair<T, T>> pairs = draw_pairs<T>(K_RANDOM_PAIRS);

            for (const Core::Layout layout : K_LAYOUTS)
            {
                SCOPED_TRACE(layout == Core::Layout::RowMajor ? "ROW_MAJOR" : "COLUMN_MAJOR");
                this->start(layout, ExecutionMode::ForceReference);
                const auto reference = this->rotate(pairs);
                this->start(layout, ExecutionMode::ForceSYCL);
                const auto device = this->rotate(pairs);

                ASSERT_EQ(device.size(), reference.size());
                for (std::size_t i = 0; i < device.size(); ++i)
                {
                    SCOPED_TRACE(::testing::Message() << "pair " << i);
                    expect_within_ulps(device[i].a, reference[i].a, "r");
                    expect_within_ulps(device[i].c, reference[i].c, "c");
                    expect_within_ulps(device[i].s, reference[i].s, "s");

                    if constexpr (Core::is_complex_v<T>)
                        EXPECT_EQ(0, std::memcmp(&device[i].b, &pairs[i].second, sizeof(T)));
                    else
                        expect_within_ulps(device[i].b, reference[i].b, "z");
                }
            }
        }

        /**
         * @brief Tests that the reference kernel forms a HALF rotation in FLOAT32 and rounds
         *        each result once.
         */
        TEST_F(Unit_RotgHalf, AHalfRotationIsComputedInFloat)
        {
            std::vector<std::pair<sycl::half, sycl::half>> pairs =
                draw_pairs<sycl::half>(K_RANDOM_HALF_PAIRS);
            pairs.emplace_back(make<sycl::half>(3.0), make<sycl::half>(4.0));

            for (const Core::Layout layout : K_LAYOUTS)
            {
                SCOPED_TRACE(engine_name(layout, ExecutionMode::ForceReference));
                start(layout, ExecutionMode::ForceReference);
                const std::vector<Rotation> rotations = rotate(pairs);

                for (std::size_t i = 0; i < rotations.size(); ++i)
                {
                    SCOPED_TRACE(::testing::Message() << "pair " << i);
                    const Kernels::GivensRotation<float> in_float =
                        Kernels::givens_rotation<StdRoot>(static_cast<float>(pairs[i].first),
                                                          static_cast<float>(pairs[i].second));

                    EXPECT_EQ(static_cast<float>(rotations[i].a),
                              static_cast<float>(sycl::half(in_float.r)));
                    EXPECT_EQ(static_cast<float>(rotations[i].b),
                              static_cast<float>(sycl::half(in_float.z)));
                    EXPECT_EQ(static_cast<float>(rotations[i].c),
                              static_cast<float>(sycl::half(in_float.c)));
                    EXPECT_EQ(static_cast<float>(rotations[i].s),
                              static_cast<float>(sycl::half(in_float.s)));
                }
            }
        }
    } // namespace Test
} // namespace SushiBLAS
