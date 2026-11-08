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
 * @file test_dispatch_complex.cpp
 * @brief Tests which elementwise and logic operations compute on COMPLEX32 and which refuse it.
 * @author Mustafa Garip
 */

#include <complex>
#include <functional>
#include <string>
#include <vector>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_comparison.hpp>
#include <SushiBLAS/tensor_expectations/tensor_fill.hpp>
using namespace SushiBLAS::Test;
using SushiBLAS::Core::DataType;
namespace sb = SushiBLAS;

using cf = std::complex<float>;

class Medium_DispatchComplexTest : public SushiBLASTest
{
protected:
    /** @brief Builds a COMPLEX32 tensor of @p values. */
    sb::Tensor complex_tensor(const std::vector<cf>& values)
    {
        return create_initialized_tensor<cf>({static_cast<int64_t>(values.size())}, values);
    }

    /** @brief Records @p record on two inputs and a COMPLEX32 result, runs it and compares the result. */
    void expect_binary(const std::function<void(sb::Tensor&, sb::Tensor&, sb::Tensor&)>& record,
                       const std::vector<cf>& av, const std::vector<cf>& bv, const std::vector<cf>& expected)
    {
        auto a = complex_tensor(av);
        auto b = complex_tensor(bv);
        auto c = engine->create_tensor({static_cast<int64_t>(expected.size())}, DataType::Complex32);
        record(a, b, c);
        recording->run().value();
        expect_tensor(c, expected);
    }

    /** @brief Records the in-place @p record on @p values, runs it and compares the tensor. */
    void expect_in_place(const std::function<void(sb::Tensor&)>& record, const std::vector<cf>& values,
                         const std::vector<cf>& expected)
    {
        auto t = complex_tensor(values);
        record(t);
        recording->run().value();
        expect_tensor(t, expected);
    }
};

TEST_F(Medium_DispatchComplexTest, AddOnComplex32)
{
    expect_binary([&](auto& a, auto& b, auto& c) { sb::ElementwiseOps(recorder()).add(a, b, c); },
                  {cf(1.0f, 2.0f), cf(3.0f, -1.0f)}, {cf(2.0f, 1.0f), cf(-1.0f, 4.0f)},
                  {cf(3.0f, 3.0f), cf(2.0f, 3.0f)});
}

TEST_F(Medium_DispatchComplexTest, SubOnComplex32)
{
    expect_binary([&](auto& a, auto& b, auto& c) { sb::ElementwiseOps(recorder()).sub(a, b, c); },
                  {cf(1.0f, 2.0f), cf(3.0f, -1.0f)}, {cf(2.0f, 1.0f), cf(-1.0f, 4.0f)},
                  {cf(-1.0f, 1.0f), cf(4.0f, -5.0f)});
}

// (1+2i)(2+i) = 5i and (3-i)(-1+4i) = 1+13i. See tests/library/umbrella/README.md.
TEST_F(Medium_DispatchComplexTest, MulOnComplex32)
{
    expect_binary([&](auto& a, auto& b, auto& c) { sb::ElementwiseOps(recorder()).mul(a, b, c); },
                  {cf(1.0f, 2.0f), cf(3.0f, -1.0f)}, {cf(2.0f, 1.0f), cf(-1.0f, 4.0f)},
                  {cf(0.0f, 5.0f), cf(1.0f, 13.0f)});
}

TEST_F(Medium_DispatchComplexTest, DivOnComplex32)
{
    expect_binary([&](auto& a, auto& b, auto& c) { sb::ElementwiseOps(recorder()).div(a, b, c); },
                  {cf(1.0f, 2.0f), cf(3.0f, -1.0f)}, {cf(2.0f, 1.0f), cf(-1.0f, 4.0f)},
                  {cf(0.8f, 0.6f), cf(-7.0f / 17.0f, -11.0f / 17.0f)});
}

TEST_F(Medium_DispatchComplexTest, NegOnComplex32)
{
    expect_in_place([&](auto& t) { sb::ElementwiseOps(recorder()).neg(t); },
                    {cf(1.0f, 2.0f), cf(-3.0f, 4.0f)}, {cf(-1.0f, -2.0f), cf(3.0f, -4.0f)});
}

// (1+2i)^2 = -3+4i ; i^2 = -1
TEST_F(Medium_DispatchComplexTest, SquareOnComplex32)
{
    expect_in_place([&](auto& t) { sb::ElementwiseOps(recorder()).square(t); },
                    {cf(1.0f, 2.0f), cf(0.0f, 1.0f)}, {cf(-3.0f, 4.0f), cf(-1.0f, 0.0f)});
}

// 1/i = -i
TEST_F(Medium_DispatchComplexTest, ReciprocalOnComplex32)
{
    expect_in_place([&](auto& t) { sb::ElementwiseOps(recorder()).reciprocal(t); },
                    {cf(2.0f, 0.0f), cf(0.0f, 1.0f)}, {cf(0.5f, 0.0f), cf(0.0f, -1.0f)});
}

// |3+4i| = 5 and |-5+12i| = 13, returned in complex storage with a zero imaginary part.
TEST_F(Medium_DispatchComplexTest, AbsOnComplex32)
{
    expect_in_place([&](auto& t) { sb::ElementwiseOps(recorder()).abs(t); },
                    {cf(3.0f, 4.0f), cf(-5.0f, 12.0f)}, {cf(5.0f, 0.0f), cf(13.0f, 0.0f)});
}

// sqrt(-9) = 3i - the principal branch, not a NaN and not a silently untouched -9.
TEST_F(Medium_DispatchComplexTest, SqrtOnComplex32)
{
    expect_in_place([&](auto& t) { sb::ElementwiseOps(recorder()).sqrt(t); },
                    {cf(4.0f, 0.0f), cf(-9.0f, 0.0f)}, {cf(2.0f, 0.0f), cf(0.0f, 3.0f)});
}

// Euler: exp(i*pi/2) = i.
TEST_F(Medium_DispatchComplexTest, ExpOnComplex32)
{
    expect_in_place([&](auto& t) { sb::ElementwiseOps(recorder()).exp(t); },
                    {cf(1.0f, 0.0f), cf(0.0f, 1.5707963f)}, {cf(2.7182818f, 0.0f), cf(0.0f, 1.0f)});
}

// log(i) = i*pi/2.
TEST_F(Medium_DispatchComplexTest, LogOnComplex32)
{
    expect_in_place([&](auto& t) { sb::ElementwiseOps(recorder()).log(t); },
                    {cf(1.0f, 0.0f), cf(0.0f, 1.0f)}, {cf(0.0f, 0.0f), cf(0.0f, 1.5707963f)});
}

namespace
{

const std::vector<cf> K_LOGIC_A = {cf(1.0f, 2.0f), cf(0.0f, 0.0f), cf(1.0f, 0.0f)};
const std::vector<cf> K_LOGIC_B = {cf(1.0f, 2.0f), cf(0.0f, 1.0f), cf(1.0f, 1.0f)};

} // namespace

// Equality is component-wise, so (1,0) != (1,1) although the real parts match.
TEST_F(Medium_DispatchComplexTest, EqualOnComplex32)
{
    expect_binary([&](auto& a, auto& b, auto& c) { sb::LogicOps(recorder()).equal(a, b, c); },
                  K_LOGIC_A, K_LOGIC_B, {cf(1.0f, 0.0f), cf(0.0f, 0.0f), cf(0.0f, 0.0f)});
}

// Truthiness is "either component nonzero": a[1] is the only zero operand.
TEST_F(Medium_DispatchComplexTest, LogicalAndOnComplex32)
{
    expect_binary([&](auto& a, auto& b, auto& c) { sb::LogicOps(recorder()).logical_and(a, b, c); },
                  K_LOGIC_A, K_LOGIC_B, {cf(1.0f, 0.0f), cf(0.0f, 0.0f), cf(1.0f, 0.0f)});
}

TEST_F(Medium_DispatchComplexTest, LogicalNotOnComplex32)
{
    auto a = complex_tensor(K_LOGIC_A);
    auto r = engine->create_tensor({3}, DataType::Complex32);
    sb::LogicOps(recorder()).logical_not(a, r);
    recording->run().value();
    expect_tensor(r, std::vector<cf>{cf(0.0f, 0.0f), cf(1.0f, 0.0f), cf(0.0f, 0.0f)});
}

TEST_F(Medium_DispatchComplexTest, WhereSelectsOnComplex32)
{
    auto cond = complex_tensor({cf(1.0f, 0.0f), cf(0.0f, 0.0f), cf(0.0f, 2.0f)});
    auto a = complex_tensor(std::vector<cf>(3, cf(7.0f, 7.0f)));
    auto b = complex_tensor(std::vector<cf>(3, cf(-1.0f, -1.0f)));
    auto out = engine->create_tensor({3}, DataType::Complex32);

    sb::LogicOps(recorder()).where(cond, a, b, out);
    recording->run().value();

    expect_tensor(out, std::vector<cf>{cf(7.0f, 7.0f), cf(-1.0f, -1.0f), cf(7.0f, 7.0f)});
}

// Operations with no complex meaning fail at record time. See tests/library/umbrella/README.md.

namespace
{

/** @brief One operation COMPLEX32 must refuse: its name and the call that records it. */
struct RefusalRow
{
    std::string name;
    std::function<void(sb::TaskRecorder&, sb::Tensor&, sb::Tensor&, sb::Tensor&)> record;
};

std::string row_name(const ::testing::TestParamInfo<RefusalRow>& info)
{
    return info.param.name;
}

} // namespace

/** @brief Records one COMPLEX32 operation per row; every row must be refused. */
class Medium_DispatchComplexRefusalTest : public SushiBLASTest, public ::testing::WithParamInterface<RefusalRow> {};

// C is not an ordered field; complex trigonometry and pow are unimplemented.
TEST_P(Medium_DispatchComplexRefusalTest, RefusesComplex32AtRecordTime)
{
    auto a = engine->create_tensor({2}, DataType::Complex32);
    auto b = engine->create_tensor({2}, DataType::Complex32);
    auto c = engine->create_tensor({2}, DataType::Complex32);

    EXPECT_THROW(GetParam().record(recorder(), a, b, c), std::invalid_argument);
}

INSTANTIATE_TEST_SUITE_P(
    Operations, Medium_DispatchComplexRefusalTest,
    ::testing::Values(
        RefusalRow{"Less", [](auto& r, auto& a, auto& b, auto& c) { sb::LogicOps(r).less(a, b, c); }},
        RefusalRow{"LessEqual", [](auto& r, auto& a, auto& b, auto& c) { sb::LogicOps(r).less_equal(a, b, c); }},
        RefusalRow{"Greater", [](auto& r, auto& a, auto& b, auto& c) { sb::LogicOps(r).greater(a, b, c); }},
        RefusalRow{"GreaterEqual",
                   [](auto& r, auto& a, auto& b, auto& c) { sb::LogicOps(r).greater_equal(a, b, c); }},
        RefusalRow{"Max", [](auto& r, auto& a, auto& b, auto& c) { sb::ElementwiseOps(r).max(a, b, c); }},
        RefusalRow{"Min", [](auto& r, auto& a, auto& b, auto& c) { sb::ElementwiseOps(r).min(a, b, c); }},
        RefusalRow{"Fmod", [](auto& r, auto& a, auto& b, auto& c) { sb::ElementwiseOps(r).fmod(a, b, c); }},
        RefusalRow{"Remainder",
                   [](auto& r, auto& a, auto& b, auto& c) { sb::ElementwiseOps(r).remainder(a, b, c); }},
        RefusalRow{"Floor", [](auto& r, auto& a, auto&, auto&) { sb::ElementwiseOps(r).floor(a); }},
        RefusalRow{"Ceil", [](auto& r, auto& a, auto&, auto&) { sb::ElementwiseOps(r).ceil(a); }},
        RefusalRow{"Round", [](auto& r, auto& a, auto&, auto&) { sb::ElementwiseOps(r).round(a); }},
        RefusalRow{"Clamp", [](auto& r, auto& a, auto&, auto&) { sb::ElementwiseOps(r).clamp(a, -1.0f, 1.0f); }},
        RefusalRow{"Sin", [](auto& r, auto& a, auto&, auto&) { sb::ElementwiseOps(r).sin(a); }},
        RefusalRow{"Cos", [](auto& r, auto& a, auto&, auto&) { sb::ElementwiseOps(r).cos(a); }},
        RefusalRow{"Tan", [](auto& r, auto& a, auto&, auto&) { sb::ElementwiseOps(r).tan(a); }},
        RefusalRow{"Pow", [](auto& r, auto& a, auto&, auto&) { sb::ElementwiseOps(r).pow(a, 2.0f); }}),
    row_name);

// max and min are Domain::RealAndIntegral and must still work on a FLOAT32 tensor.
TEST_F(Medium_DispatchComplexTest, MaxOnFloat32IsUnaffected)
{
    auto a = engine->create_tensor({4});
    auto b = engine->create_tensor({4});
    auto c = engine->create_tensor({4});
    fill_tensor(a, {1.0f, 2.0f, -3.0f, 0.5f});
    fill_tensor(b, {2.0f, 0.5f, 4.0f, -1.5f});

    sb::ElementwiseOps(recorder()).max(a, b, c);
    recording->run().value();
    expect_tensor_equal(c, {2.0f, 2.0f, 4.0f, 0.5f});
}

TEST_F(Medium_DispatchComplexTest, MinOnFloat32IsUnaffected)
{
    auto a = engine->create_tensor({4});
    auto b = engine->create_tensor({4});
    auto c = engine->create_tensor({4});
    fill_tensor(a, {1.0f, 2.0f, -3.0f, 0.5f});
    fill_tensor(b, {2.0f, 0.5f, 4.0f, -1.5f});

    sb::ElementwiseOps(recorder()).min(a, b, c);
    recording->run().value();
    expect_tensor_equal(c, {1.0f, 0.5f, -3.0f, -1.5f});
}
