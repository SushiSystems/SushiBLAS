/**************************************************************************/
/* test_complex_nonlinear.cpp                                             */
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
 * @file test_complex_nonlinear.cpp
 * @brief Tests complex sigmoid, tanh, sqrt and log against hand values, and elu's refusal.
 * @author Mustafa Garip
 */

#include <cmath>
#include <limits>
#include <complex>
#include <stdexcept>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"

using namespace SushiBLAS::Test;

namespace
{
    /// @brief Absolute per-component tolerance for COMPLEX32 results of magnitude near one.
    constexpr double K_COMPLEX32_TOLERANCE = 1e-5;

    /// @brief Absolute per-component tolerance for COMPLEX64 results of magnitude near one.
    constexpr double K_COMPLEX64_TOLERANCE = 1e-12;

    /// @brief ELU slope applied to inputs whose real part is not positive.
    constexpr double K_ELU_ALPHA = 1.0;

    /**
     * @brief Returns the five probe inputs (0,0), (1,0), (0,1), (1,1) and (-2,0.5).
     */
    template<typename T>
    std::vector<std::complex<T>> probe_inputs()
    {
        return {std::complex<T>(0, 0), std::complex<T>(1, 0), std::complex<T>(0, 1),
                std::complex<T>(1, 1), std::complex<T>(-2, 0.5)};
    }

    /**
     * @brief Returns the per-component tolerance for the complex element type T.
     */
    template<typename T>
    constexpr double tolerance_for()
    {
        return sizeof(T) == sizeof(float) ? K_COMPLEX32_TOLERANCE : K_COMPLEX64_TOLERANCE;
    }

    /**
     * @brief Applies @p reference to every probe input on the host.
     */
    template<typename T, typename Reference>
    std::vector<std::complex<T>> host_expected(Reference reference)
    {
        std::vector<std::complex<T>> expected;
        for (const std::complex<T>& x : probe_inputs<T>())
            expected.push_back(reference(x));
        return expected;
    }
} // namespace

/**
 * @brief Typed fixture; T is the real component type of the complex element.
 */
template<typename T>
class Unit_ComplexNonlinearTest : public SushiBLASTest
{

};

using ComplexComponentTypes = ::testing::Types<float, double>;
SB_TYPED_TEST_SUITE(Unit_ComplexNonlinearTest, ComplexComponentTypes);

TYPED_TEST(Unit_ComplexNonlinearTest, SigmoidMatchesHostReference)
{
    using C = std::complex<TypeParam>;
    auto t = this->template create_initialized_tensor<C>({5}, probe_inputs<TypeParam>());

    SushiBLAS::NonLinearOps(*this->engine).sigmoid(t);
    this->engine->execute();

    auto expected = this->template create_initialized_tensor<C>(
        {5}, host_expected<TypeParam>([](const C& x) { return C(1) / (C(1) + std::exp(-x)); }));
    SB_ASSERT_TENSOR_NEAR(t, expected, tolerance_for<TypeParam>(), C);
}

TYPED_TEST(Unit_ComplexNonlinearTest, TanhMatchesHostReference)
{
    using C = std::complex<TypeParam>;
    auto t = this->template create_initialized_tensor<C>({5}, probe_inputs<TypeParam>());

    SushiBLAS::NonLinearOps(*this->engine).tanh(t);
    this->engine->execute();

    auto expected = this->template create_initialized_tensor<C>(
        {5}, host_expected<TypeParam>([](const C& x) { return std::tanh(x); }));
    SB_ASSERT_TENSOR_NEAR(t, expected, tolerance_for<TypeParam>(), C);
}

TYPED_TEST(Unit_ComplexNonlinearTest, EluRefusesComplex)
{
    using C = std::complex<TypeParam>;
    auto t = this->template create_initialized_tensor<C>({5}, probe_inputs<TypeParam>());

    EXPECT_THROW(SushiBLAS::NonLinearOps(*this->engine).elu(t, K_ELU_ALPHA), std::runtime_error);
}

/// @brief Runs the complex sqrt and log edge cases.
class Unit_ComplexSqrtLogTest : public SushiBLASTest
{

};

/**
 * @brief Tests that sqrt of (x, +inf) is (+inf, +inf) for a positive and a negative x.
 */
TEST_F(Unit_ComplexSqrtLogTest, SqrtOfInfiniteImaginaryPartIsInfinite)
{
    using C = std::complex<double>;
    const double inf = std::numeric_limits<double>::infinity();
    auto t = create_initialized_tensor<C>({2}, {C(4.0, inf), C(-4.0, inf)});

    SushiBLAS::ElementwiseOps(*engine).sqrt(t);
    engine->execute();

    const std::vector<C> got = read_tensor<C>(t);
    for (const C& z : got)
    {
        EXPECT_TRUE(std::isinf(z.real()) && z.real() > 0.0);
        EXPECT_TRUE(std::isinf(z.imag()) && z.imag() > 0.0);
    }
}

/**
 * @brief Tests that log of (1, 1e-4) keeps the real part 0.5 * log1p(1e-8) to full precision.
 */
TEST_F(Unit_ComplexSqrtLogTest, LogNearTheUnitCircleKeepsItsRealPart)
{
    using C = std::complex<double>;
    auto t = create_initialized_tensor<C>({1}, {C(1.0, 1e-4)});

    SushiBLAS::ElementwiseOps(*engine).log(t);
    engine->execute();

    const C got = read_tensor<C>(t)[0];
    EXPECT_NEAR(5.0e-9 - 2.5e-17, got.real(), 1e-22);
    EXPECT_NEAR(1e-4 - 1e-12 / 3.0, got.imag(), 1e-18);
}
