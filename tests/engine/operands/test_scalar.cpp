/**************************************************************************/
/* test_scalar.cpp                                                        */
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
 * @file test_scalar.cpp
 * @brief Tests BLASScalar: what converts to it, what it narrows to and what it reports.
 * @author Mustafa Garip
 */

#include <cmath>
#include <string>
#include <vector>
#include <complex>
#include <limits>
#include <type_traits>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/operands/scalar.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief The largest finite HALF value, which binary32 holds exactly.
        constexpr double K_LARGEST_HALF = 65504.0;

        /// @brief A value HALF cannot hold exactly; its neighbours in HALF are 2^-14 apart.
        constexpr double K_INEXACT_IN_HALF = 0.1;

        /// @brief The distance between the HALF values around K_INEXACT_IN_HALF.
        constexpr double K_HALF_SPACING_AT_INEXACT = 1.0 / 16384.0;

        /// @brief A value below half of the smallest positive HALF, 2^-24.
        constexpr double K_BELOW_HALF_RANGE = 1e-8;

        /// @brief The smallest positive HALF value.
        constexpr double K_SMALLEST_HALF = 1.0 / 16777216.0;

        namespace
        {
            /** @brief Names a type and whether it converts to a BLASScalar without a cast. */
            struct ConversionRow
            {
                const char* name;  /**< The row name a failure shows. */
                bool converts;     /**< What std::is_convertible_v reports for the type. */
            };

            /** @brief Returns one row per type that converts implicitly. */
            std::vector<ConversionRow> conversion_rows()
            {
                return {{"Float", std::is_convertible_v<float, BLASScalar>},
                        {"Int", std::is_convertible_v<int, BLASScalar>},
                        {"Double", std::is_convertible_v<double, BLASScalar>},
                        {"ComplexDouble", std::is_convertible_v<std::complex<double>, BLASScalar>}};
            }
        } // namespace

        /** @brief Runs one row of conversion_rows(). */
        class Small_BlasScalarConversion : public ::testing::TestWithParam<ConversionRow>
        {
        };

        TEST_P(Small_BlasScalarConversion, RealAndComplexDoubleConvertImplicitly)
        {
            EXPECT_TRUE(GetParam().converts);
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_BlasScalarConversion, ::testing::ValuesIn(conversion_rows()),
            [](const ::testing::TestParamInfo<ConversionRow>& info)
            { return std::string(info.param.name); });

        TEST(Small_BlasScalar, ComplexFloatDoesNotConvertImplicitly)
        {
            EXPECT_FALSE((std::is_convertible_v<std::complex<float>, BLASScalar>));
        }

        namespace
        {
            /** @brief Names a real scalar and the real part it must keep. */
            struct RealValueRow
            {
                const char* name;    /**< The row name a failure shows. */
                BLASScalar scalar;   /**< The scalar built from a real input. */
                double expected;     /**< The real part the scalar keeps. */
            };

            /** @brief Returns one row per real input type. */
            std::vector<RealValueRow> real_value_rows()
            {
                return {{"FromFloat", BLASScalar(1.5f), 1.5},
                        {"FromInt", BLASScalar(3), 3.0},
                        {"FromDouble", BLASScalar(-2.25), -2.25}};
            }
        } // namespace

        /** @brief Runs one row of real_value_rows(). */
        class Small_BlasScalarRealValue : public ::testing::TestWithParam<RealValueRow>
        {
        };

        TEST_P(Small_BlasScalarRealValue, ARealValueKeepsItsValueAndAZeroImaginaryPart)
        {
            EXPECT_EQ(GetParam().scalar.real(), GetParam().expected);
            EXPECT_EQ(GetParam().scalar.imag(), 0.0);
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_BlasScalarRealValue, ::testing::ValuesIn(real_value_rows()),
            [](const ::testing::TestParamInfo<RealValueRow>& info)
            { return std::string(info.param.name); });

        TEST(Small_BlasScalar, AComplexValueKeepsBothParts)
        {
            const BLASScalar scalar = std::complex<double>(1.5, -2.25);

            EXPECT_EQ(scalar.real(), 1.5);
            EXPECT_EQ(scalar.imag(), -2.25);
        }

        namespace
        {
            /** @brief Names one read of the scalar 1.5 in a real type, widened to double. */
            struct RealTypeRow
            {
                const char* name;                       /**< The row name a failure shows. */
                double (*read)(const BLASScalar&);      /**< Reads the scalar in the type. */
            };

            /** @brief Returns one row per real type. */
            std::vector<RealTypeRow> real_type_rows()
            {
                return {{"Float", [](const BLASScalar& s) { return static_cast<double>(s.as<float>()); }},
                        {"Double", [](const BLASScalar& s) { return s.as<double>(); }},
                        {"Half",
                         [](const BLASScalar& s)
                         { return static_cast<double>(static_cast<float>(s.as<sycl::half>())); }}};
            }
        } // namespace

        /** @brief Runs one row of real_type_rows(). */
        class Small_BlasScalarRealType : public ::testing::TestWithParam<RealTypeRow>
        {
        };

        TEST_P(Small_BlasScalarRealType, AsReturnsTheValueInTheType)
        {
            EXPECT_EQ(GetParam().read(BLASScalar(1.5)), 1.5);
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_BlasScalarRealType, ::testing::ValuesIn(real_type_rows()),
            [](const ::testing::TestParamInfo<RealTypeRow>& info)
            { return std::string(info.param.name); });

        TEST(Small_BlasScalar, AsFloatRoundsADoubleOnce)
        {
            const BLASScalar scalar = K_INEXACT_IN_HALF;

            EXPECT_EQ(scalar.as<float>(), static_cast<float>(K_INEXACT_IN_HALF));
        }

        TEST(Small_BlasScalar, AsDoubleKeepsTheValue)
        {
            const BLASScalar scalar = K_INEXACT_IN_HALF;

            EXPECT_EQ(scalar.as<double>(), K_INEXACT_IN_HALF);
        }

        namespace
        {
            /** @brief Names one read of a scalar in a complex type, widened to complex double. */
            struct ComplexTypeRow
            {
                const char* name;                                   /**< The row name a failure shows. */
                std::complex<double> (*read)(const BLASScalar&);    /**< Reads the scalar in the type. */
            };

            /** @brief Returns one row per complex type. */
            std::vector<ComplexTypeRow> complex_type_rows()
            {
                return {{"ComplexFloat",
                         [](const BLASScalar& s)
                         { return std::complex<double>(s.as<std::complex<float>>()); }},
                        {"ComplexDouble",
                         [](const BLASScalar& s) { return s.as<std::complex<double>>(); }}};
            }
        } // namespace

        /** @brief Runs one row of complex_type_rows(). */
        class Small_BlasScalarComplexType : public ::testing::TestWithParam<ComplexTypeRow>
        {
        };

        TEST_P(Small_BlasScalarComplexType, AsReturnsBothPartsInTheType)
        {
            EXPECT_EQ(GetParam().read(BLASScalar(std::complex<double>(1.5, -2.25))),
                      std::complex<double>(1.5, -2.25));
        }

        TEST_P(Small_BlasScalarComplexType, AsGivesARealScalarAZeroImaginaryPart)
        {
            EXPECT_EQ(GetParam().read(BLASScalar(1.5)), std::complex<double>(1.5, 0.0));
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_BlasScalarComplexType, ::testing::ValuesIn(complex_type_rows()),
            [](const ::testing::TestParamInfo<ComplexTypeRow>& info)
            { return std::string(info.param.name); });

        namespace
        {
            /** @brief Names a value that as<half> must round exactly as a conversion through float does. */
            struct HalfValueRow
            {
                const char* name; /**< The row name a failure shows. */
                double value;     /**< The scalar converted. */
            };

            /** @brief Returns one row per value. */
            std::vector<HalfValueRow> half_value_rows()
            {
                return {{"Inexact", K_INEXACT_IN_HALF},
                        {"Largest", K_LARGEST_HALF},
                        {"BelowRange", K_BELOW_HALF_RANGE}};
            }
        } // namespace

        /** @brief Runs one row of half_value_rows(). */
        class Small_BlasScalarHalfValue : public ::testing::TestWithParam<HalfValueRow>
        {
        };

        TEST_P(Small_BlasScalarHalfValue, AsHalfRoundsThroughFloat)
        {
            const BLASScalar scalar = GetParam().value;
            const sycl::half through_float = sycl::half(static_cast<float>(GetParam().value));

            EXPECT_EQ(static_cast<float>(scalar.as<sycl::half>()),
                      static_cast<float>(through_float));
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_BlasScalarHalfValue, ::testing::ValuesIn(half_value_rows()),
            [](const ::testing::TestParamInfo<HalfValueRow>& info)
            { return std::string(info.param.name); });

        TEST(Small_BlasScalar, AsHalfKeepsTheLargestHalfExactly)
        {
            const BLASScalar scalar = K_LARGEST_HALF;

            EXPECT_EQ(static_cast<double>(static_cast<float>(scalar.as<sycl::half>())),
                      K_LARGEST_HALF);
        }

        TEST(Small_BlasScalar, AsHalfLandsOnANeighbourOfAnInexactValue)
        {
            const BLASScalar scalar = K_INEXACT_IN_HALF;
            const double rounded = static_cast<double>(static_cast<float>(scalar.as<sycl::half>()));

            EXPECT_LE(std::fabs(rounded - K_INEXACT_IN_HALF), K_HALF_SPACING_AT_INEXACT);
        }

        TEST(Small_BlasScalar, AsHalfTakesAValueBelowTheRangeToZeroOrTheSmallestHalf)
        {
            const BLASScalar scalar = K_BELOW_HALF_RANGE;
            const double rounded = static_cast<double>(static_cast<float>(scalar.as<sycl::half>()));

            EXPECT_GE(rounded, 0.0);
            EXPECT_LE(rounded, K_SMALLEST_HALF);
        }

        TEST(Small_BlasScalar, IsRealHoldsForAZeroImaginaryPart)
        {
            EXPECT_TRUE(BLASScalar(2.0).is_real());
            EXPECT_TRUE(BLASScalar(std::complex<double>(2.0, 0.0)).is_real());
            EXPECT_TRUE(BLASScalar(std::complex<double>(2.0, -0.0)).is_real());
            EXPECT_FALSE(BLASScalar(std::complex<double>(2.0, 1e-300)).is_real());
        }

        TEST(Small_BlasScalar, IsRealFailsForANanImaginaryPart)
        {
            const double nan = std::numeric_limits<double>::quiet_NaN();

            EXPECT_FALSE(BLASScalar(std::complex<double>(1.0, nan)).is_real());
        }

        TEST(Small_BlasScalar, IsZeroHoldsOnlyWhenBothPartsAreZero)
        {
            EXPECT_TRUE(BLASScalar(0.0).is_zero());
            EXPECT_TRUE(BLASScalar(-0.0).is_zero());
            EXPECT_TRUE(BLASScalar(std::complex<double>(0.0, 0.0)).is_zero());
            EXPECT_FALSE(BLASScalar(1e-300).is_zero());
            EXPECT_FALSE(BLASScalar(std::complex<double>(0.0, 1e-300)).is_zero());
        }

        TEST(Small_BlasScalar, IsZeroFailsForANanPart)
        {
            const double nan = std::numeric_limits<double>::quiet_NaN();

            EXPECT_FALSE(BLASScalar(nan).is_zero());
            EXPECT_FALSE(BLASScalar(std::complex<double>(0.0, nan)).is_zero());
        }
    } // namespace Test
} // namespace SushiBLAS
