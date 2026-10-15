/**************************************************************************/
/* test_dispatch_complex.cpp                                              */
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
 * @file test_dispatch_complex.cpp
 * @brief Tests which elementwise and logic operations compute on COMPLEX32 and which refuse it.
 * @author Mustafa Garip
 */

#include <complex>
#include <vector>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_comparison.hpp>
#include <SushiBLAS/tensor_expectations/tensor_fill.hpp>
using namespace SushiBLAS::Test;
using SushiBLAS::Core::DataType;

using cf = std::complex<float>;

class Unit_DispatchComplexTest : public SushiBLASTest {};

TEST_F(Unit_DispatchComplexTest, BinaryArithmeticOnComplex32)
{
    const std::vector<cf> av = {cf(1.0f, 2.0f), cf(3.0f, -1.0f)};
    const std::vector<cf> bv = {cf(2.0f, 1.0f), cf(-1.0f, 4.0f)};

    auto a = create_initialized_tensor<cf>({2}, av);
    auto b = create_initialized_tensor<cf>({2}, bv);
    auto c = engine->create_tensor({2}, DataType::Complex32);

    SushiBLAS::ElementwiseOps(*engine).add(a, b, c);
    engine->execute().value();
    auto add_expected = create_initialized_tensor<cf>({2}, {cf(3.0f, 3.0f), cf(2.0f, 3.0f)});
    SB_ASSERT_TENSOR_NEAR(c, add_expected, 1e-4, cf);

    SushiBLAS::ElementwiseOps(*engine).sub(a, b, c);
    engine->execute().value();
    auto sub_expected = create_initialized_tensor<cf>({2}, {cf(-1.0f, 1.0f), cf(4.0f, -5.0f)});
    SB_ASSERT_TENSOR_NEAR(c, sub_expected, 1e-4, cf);

    // (1+2i)(2+i) = 5i and (3-i)(-1+4i) = 1+13i. See tests/library/sushiblas/README.md.
    SushiBLAS::ElementwiseOps(*engine).mul(a, b, c);
    engine->execute().value();
    auto mul_expected = create_initialized_tensor<cf>({2}, {cf(0.0f, 5.0f), cf(1.0f, 13.0f)});
    SB_ASSERT_TENSOR_NEAR(c, mul_expected, 1e-4, cf);

    SushiBLAS::ElementwiseOps(*engine).div(a, b, c);
    engine->execute().value();
    auto div_expected = create_initialized_tensor<cf>({2}, {cf(0.8f, 0.6f), cf(-7.0f / 17.0f, -11.0f / 17.0f)});
    SB_ASSERT_TENSOR_NEAR(c, div_expected, 1e-4, cf);
}

TEST_F(Unit_DispatchComplexTest, UnaryArithmeticOnComplex32)
{
    auto neg = create_initialized_tensor<cf>({2}, {cf(1.0f, 2.0f), cf(-3.0f, 4.0f)});
    SushiBLAS::ElementwiseOps(*engine).neg(neg);
    engine->execute().value();
    auto neg_expected = create_initialized_tensor<cf>({2}, {cf(-1.0f, -2.0f), cf(3.0f, -4.0f)});
    SB_ASSERT_TENSOR_NEAR(neg, neg_expected, 1e-4, cf);

    // (1+2i)^2 = -3+4i ; i^2 = -1
    auto square = create_initialized_tensor<cf>({2}, {cf(1.0f, 2.0f), cf(0.0f, 1.0f)});
    SushiBLAS::ElementwiseOps(*engine).square(square);
    engine->execute().value();
    auto square_expected = create_initialized_tensor<cf>({2}, {cf(-3.0f, 4.0f), cf(-1.0f, 0.0f)});
    SB_ASSERT_TENSOR_NEAR(square, square_expected, 1e-4, cf);

    // 1/i = -i
    auto recip = create_initialized_tensor<cf>({2}, {cf(2.0f, 0.0f), cf(0.0f, 1.0f)});
    SushiBLAS::ElementwiseOps(*engine).reciprocal(recip);
    engine->execute().value();
    auto recip_expected = create_initialized_tensor<cf>({2}, {cf(0.5f, 0.0f), cf(0.0f, -1.0f)});
    SB_ASSERT_TENSOR_NEAR(recip, recip_expected, 1e-4, cf);
}

TEST_F(Unit_DispatchComplexTest, TranscendentalsOnComplex32)
{
    // |3+4i| = 5 and |-5+12i| = 13, returned in complex storage with a zero imaginary part.
    auto abs_t = create_initialized_tensor<cf>({2}, {cf(3.0f, 4.0f), cf(-5.0f, 12.0f)});
    SushiBLAS::ElementwiseOps(*engine).abs(abs_t);
    engine->execute().value();
    auto abs_expected = create_initialized_tensor<cf>({2}, {cf(5.0f, 0.0f), cf(13.0f, 0.0f)});
    SB_ASSERT_TENSOR_NEAR(abs_t, abs_expected, 1e-4, cf);

    // sqrt(-9) = 3i - the principal branch, not a NaN and not a silently untouched -9.
    auto sqrt_t = create_initialized_tensor<cf>({2}, {cf(4.0f, 0.0f), cf(-9.0f, 0.0f)});
    SushiBLAS::ElementwiseOps(*engine).sqrt(sqrt_t);
    engine->execute().value();
    auto sqrt_expected = create_initialized_tensor<cf>({2}, {cf(2.0f, 0.0f), cf(0.0f, 3.0f)});
    SB_ASSERT_TENSOR_NEAR(sqrt_t, sqrt_expected, 1e-4, cf);

    // Euler: exp(i*pi/2) = i.
    auto exp_t = create_initialized_tensor<cf>({2}, {cf(1.0f, 0.0f), cf(0.0f, 1.5707963f)});
    SushiBLAS::ElementwiseOps(*engine).exp(exp_t);
    engine->execute().value();
    auto exp_expected = create_initialized_tensor<cf>({2}, {cf(2.7182818f, 0.0f), cf(0.0f, 1.0f)});
    SB_ASSERT_TENSOR_NEAR(exp_t, exp_expected, 1e-4, cf);

    // log(i) = i*pi/2.
    auto log_t = create_initialized_tensor<cf>({2}, {cf(1.0f, 0.0f), cf(0.0f, 1.0f)});
    SushiBLAS::ElementwiseOps(*engine).log(log_t);
    engine->execute().value();
    auto log_expected = create_initialized_tensor<cf>({2}, {cf(0.0f, 0.0f), cf(0.0f, 1.5707963f)});
    SB_ASSERT_TENSOR_NEAR(log_t, log_expected, 1e-4, cf);
}

TEST_F(Unit_DispatchComplexTest, EqualityAndConnectivesOnComplex32)
{
    auto a = create_initialized_tensor<cf>({3}, {cf(1.0f, 2.0f), cf(0.0f, 0.0f), cf(1.0f, 0.0f)});
    auto b = create_initialized_tensor<cf>({3}, {cf(1.0f, 2.0f), cf(0.0f, 1.0f), cf(1.0f, 1.0f)});
    auto r = engine->create_tensor({3}, DataType::Complex32);

    // Equality is component-wise, so (1,0) != (1,1) although the real parts match.
    SushiBLAS::LogicOps(*engine).equal(a, b, r);
    engine->execute().value();
    auto eq_expected = create_initialized_tensor<cf>({3}, {cf(1.0f, 0.0f), cf(0.0f, 0.0f), cf(0.0f, 0.0f)});
    SB_ASSERT_TENSOR_NEAR(r, eq_expected, 1e-4, cf);

    // Truthiness is "either component nonzero": a[1] is the only zero operand.
    SushiBLAS::LogicOps(*engine).logical_and(a, b, r);
    engine->execute().value();
    auto and_expected = create_initialized_tensor<cf>({3}, {cf(1.0f, 0.0f), cf(0.0f, 0.0f), cf(1.0f, 0.0f)});
    SB_ASSERT_TENSOR_NEAR(r, and_expected, 1e-4, cf);

    SushiBLAS::LogicOps(*engine).logical_not(a, r);
    engine->execute().value();
    auto not_expected = create_initialized_tensor<cf>({3}, {cf(0.0f, 0.0f), cf(1.0f, 0.0f), cf(0.0f, 0.0f)});
    SB_ASSERT_TENSOR_NEAR(r, not_expected, 1e-4, cf);
}

TEST_F(Unit_DispatchComplexTest, WhereSelectsOnComplex32)
{
    auto cond = create_initialized_tensor<cf>({3}, {cf(1.0f, 0.0f), cf(0.0f, 0.0f), cf(0.0f, 2.0f)});
    auto a = create_initialized_tensor<cf>({3}, {cf(7.0f, 7.0f), cf(7.0f, 7.0f), cf(7.0f, 7.0f)});
    auto b = create_initialized_tensor<cf>({3}, {cf(-1.0f, -1.0f), cf(-1.0f, -1.0f), cf(-1.0f, -1.0f)});
    auto out = engine->create_tensor({3}, DataType::Complex32);

    SushiBLAS::LogicOps(*engine).where(cond, a, b, out);
    engine->execute().value();

    auto expected = create_initialized_tensor<cf>({3}, {cf(7.0f, 7.0f), cf(-1.0f, -1.0f), cf(7.0f, 7.0f)});
    SB_ASSERT_TENSOR_NEAR(out, expected, 1e-4, cf);
}

// Operations with no complex meaning fail at record time. See tests/library/sushiblas/README.md.

TEST_F(Unit_DispatchComplexTest, OrderingComparisonsRejectComplex32)
{
    auto a = engine->create_tensor({2}, DataType::Complex32);
    auto b = engine->create_tensor({2}, DataType::Complex32);
    auto r = engine->create_tensor({2}, DataType::Complex32);

    // C is not an ordered field. See tests/library/sushiblas/README.md.
    EXPECT_THROW(SushiBLAS::LogicOps(*engine).less(a, b, r), std::invalid_argument);
    EXPECT_THROW(SushiBLAS::LogicOps(*engine).less_equal(a, b, r), std::invalid_argument);
    EXPECT_THROW(SushiBLAS::LogicOps(*engine).greater(a, b, r), std::invalid_argument);
    EXPECT_THROW(SushiBLAS::LogicOps(*engine).greater_equal(a, b, r), std::invalid_argument);
}

TEST_F(Unit_DispatchComplexTest, OrderDependentElementwiseOpsRejectComplex32)
{
    auto a = engine->create_tensor({2}, DataType::Complex32);
    auto b = engine->create_tensor({2}, DataType::Complex32);
    auto c = engine->create_tensor({2}, DataType::Complex32);

    EXPECT_THROW(SushiBLAS::ElementwiseOps(*engine).max(a, b, c), std::invalid_argument);
    EXPECT_THROW(SushiBLAS::ElementwiseOps(*engine).min(a, b, c), std::invalid_argument);
    EXPECT_THROW(SushiBLAS::ElementwiseOps(*engine).fmod(a, b, c), std::invalid_argument);
    EXPECT_THROW(SushiBLAS::ElementwiseOps(*engine).remainder(a, b, c), std::invalid_argument);
}

TEST_F(Unit_DispatchComplexTest, RoundingAndClampRejectComplex32)
{
    auto t = engine->create_tensor({2}, DataType::Complex32);

    EXPECT_THROW(SushiBLAS::ElementwiseOps(*engine).floor(t), std::invalid_argument);
    EXPECT_THROW(SushiBLAS::ElementwiseOps(*engine).ceil(t), std::invalid_argument);
    EXPECT_THROW(SushiBLAS::ElementwiseOps(*engine).round(t), std::invalid_argument);
    EXPECT_THROW(SushiBLAS::ElementwiseOps(*engine).clamp(t, -1.0f, 1.0f), std::invalid_argument);
}

TEST_F(Unit_DispatchComplexTest, RemainingRealOnlyOpsRejectComplex32)
{
    // Complex trigonometry and pow are not implemented. See tests/library/sushiblas/README.md.
    auto t = engine->create_tensor({2}, DataType::Complex32);

    EXPECT_THROW(SushiBLAS::ElementwiseOps(*engine).sin(t), std::invalid_argument);
    EXPECT_THROW(SushiBLAS::ElementwiseOps(*engine).cos(t), std::invalid_argument);
    EXPECT_THROW(SushiBLAS::ElementwiseOps(*engine).tan(t), std::invalid_argument);
    EXPECT_THROW(SushiBLAS::ElementwiseOps(*engine).pow(t, 2.0f), std::invalid_argument);
}

TEST_F(Unit_DispatchComplexTest, RealDtypesAreUnaffected)
{
    // max and min are Domain::RealAndIntegral and must still work on a FLOAT32 tensor.
    auto a = engine->create_tensor({4});
    auto b = engine->create_tensor({4});
    auto c = engine->create_tensor({4});
    fill_tensor(a, {1.0f, 2.0f, -3.0f, 0.5f});
    fill_tensor(b, {2.0f, 0.5f, 4.0f, -1.5f});

    SushiBLAS::ElementwiseOps(*engine).max(a, b, c);
    engine->execute().value();
    verify_tensor(c, {2.0f, 2.0f, 4.0f, 0.5f});

    SushiBLAS::ElementwiseOps(*engine).min(a, b, c);
    engine->execute().value();
    verify_tensor(c, {1.0f, 0.5f, -3.0f, -1.5f});
}
