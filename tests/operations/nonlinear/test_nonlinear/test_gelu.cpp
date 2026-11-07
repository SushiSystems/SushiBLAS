/**************************************************************************/
/* test_gelu.cpp                                                          */
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
 * @file test_gelu.cpp
 * @brief Tests NonLinearOps::gelu and gelu_backward against the tanh form the header documents,
 *        evaluated on the host.
 * @author Mustafa Garip
 */

#include <cmath>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_comparison.hpp>
#include <SushiBLAS/tensor_expectations/tensor_fill.hpp>
using namespace SushiBLAS::Test;

namespace
{
    /// @brief The inputs both tests evaluate GELU at.
    const std::vector<float> K_INPUTS{-1.0f, 0.0f, 1.0f, 2.0f};

    /// @brief sqrt(2 / pi), the factor of the tanh form of GELU.
    constexpr double K_SQRT_TWO_OVER_PI = 0.7978845608028654;

    /// @brief The cubic coefficient of the tanh form of GELU.
    constexpr double K_CUBIC = 0.044715;

    /// @brief Returns 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))), the form gelu documents.
    double gelu_on_the_host(double x)
    {
        return 0.5 * x * (1.0 + std::tanh(K_SQRT_TWO_OVER_PI * (x + K_CUBIC * x * x * x)));
    }

    /**
     * @brief Returns the derivative of gelu_on_the_host: the cdf 0.5 * (1 + t) plus the term
     *        0.5 * x * (1 - t^2) * d(inner)/dx, where t is the tanh of the inner argument.
     */
    double gelu_derivative_on_the_host(double x)
    {
        const double t = std::tanh(K_SQRT_TWO_OVER_PI * (x + K_CUBIC * x * x * x));
        const double d_inner = K_SQRT_TWO_OVER_PI * (1.0 + 3.0 * K_CUBIC * x * x);
        return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * d_inner;
    }

    /// @brief Returns f(x) in float for every input, f a host function of a double.
    template<typename Function>
    std::vector<float> on_the_host(Function f)
    {
        std::vector<float> values;
        for (const float x : K_INPUTS)
            values.push_back(static_cast<float>(f(static_cast<double>(x))));
        return values;
    }
} // namespace

class Medium_GELUTest : public SushiBLASTest {};

TEST_F(Medium_GELUTest, ForwardIsTheTanhFormTheHeaderDocuments)
{
    auto t = engine->create_tensor({4});
    fill_tensor(t, K_INPUTS);

    SushiBLAS::NonLinearOps(recorder()).gelu(t);
    recording->run().value();

    expect_tensor(t, on_the_host(gelu_on_the_host));
}

TEST_F(Medium_GELUTest, BackwardIsTheDerivativeOfTheTanhFormTimesTheGradient)
{
    auto dy = engine->create_tensor({4});
    auto x = engine->create_tensor({4});
    auto dx = engine->create_tensor({4});

    fill_tensor(dy, {1.0f, 1.0f, 1.0f, 1.0f});
    fill_tensor(x, K_INPUTS);
    fill_tensor(dx, {0.0f, 0.0f, 0.0f, 0.0f});

    SushiBLAS::NonLinearOps(recorder()).gelu_backward(dy, x, dx);
    recording->run().value();

    expect_tensor(dx, on_the_host(gelu_derivative_on_the_host));
}
