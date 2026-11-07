/**************************************************************************/
/* test_initializers.cpp                                                  */
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
 * @file test_initializers.cpp
 * @brief Tests the Xavier and He initializers of RandomOps.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine_fixture/host_engine_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_readback.hpp>
#include "../random_statistics.hpp"
using namespace SushiBLAS::Test;
using SushiBLAS::RandomOps;
using SushiBLAS::Tensor;

namespace
{
    /// @brief Holds the side of the square tensor each initializer fills: 256 * 256 draws.
    constexpr int64_t K_SIDE = 256;

    /// @brief Holds the fan-in of the initializers.
    constexpr int64_t K_FAN_IN = K_SIDE;

    /// @brief Holds the fan-out of the Xavier initializers.
    constexpr int64_t K_FAN_OUT = K_SIDE;

    /// @brief Holds the seed the rows share.
    constexpr uint64_t K_SEED = 20261008;

    /// @brief Holds an infinite bound, for a law whose draws are unbounded.
    constexpr double K_UNBOUNDED = std::numeric_limits<double>::infinity();

    /** @brief Records one initializer on a tensor. */
    using Record = std::function<void(RandomOps&, Tensor&)>;

    /** @brief Names one initializer, the law it draws from and the half width it stays inside. */
    struct InitializerRow
    {
        const char* name; /**< The row name a failure shows. */
        Record record;    /**< Records the initializer. */
        Law law;          /**< The law of the draws. */
        double limit;     /**< The draws lie in [-limit, limit]; infinite when unbounded. */
    };

    /** @brief Returns the bound as the float the kernel compares against. */
    double float_bound(double limit)
    {
        return static_cast<double>(static_cast<float>(limit));
    }

    /** @brief Returns the bound sqrt(6 / fan) of a uniform initializer. */
    double uniform_limit(int64_t fan)
    {
        return float_bound(std::sqrt(6.0 / static_cast<double>(fan)));
    }

    /** @brief Returns one row per initializer. */
    std::vector<InitializerRow> initializer_rows()
    {
        const double xavier_limit = uniform_limit(K_FAN_IN + K_FAN_OUT);
        const double he_limit = uniform_limit(K_FAN_IN);
        const double xavier_stddev = std::sqrt(2.0 / static_cast<double>(K_FAN_IN + K_FAN_OUT));
        const double he_stddev = std::sqrt(2.0 / static_cast<double>(K_FAN_IN));

        return {{"XavierUniform",
                 [](RandomOps& r, Tensor& t) { r.xavier_uniform(t, K_FAN_IN, K_FAN_OUT); },
                 uniform_law(-xavier_limit, xavier_limit), xavier_limit},
                {"XavierNormal",
                 [](RandomOps& r, Tensor& t) { r.xavier_normal(t, K_FAN_IN, K_FAN_OUT); },
                 normal_law(0.0, xavier_stddev), K_UNBOUNDED},
                {"HeUniform", [](RandomOps& r, Tensor& t) { r.he_uniform(t, K_FAN_IN); },
                 uniform_law(-he_limit, he_limit), he_limit},
                {"HeNormal", [](RandomOps& r, Tensor& t) { r.he_normal(t, K_FAN_IN); },
                 normal_law(0.0, he_stddev), K_UNBOUNDED}};
    }
}

/** @brief Runs one row of initializer_rows() on a FLOAT32 square tensor. */
class Small_InitializersTest : public HostEngineTest,
                               public ::testing::WithParamInterface<InitializerRow>
{
    protected:
        /** @brief Records the initializer of the row from K_SEED, runs it and returns the elements widened. */
        std::vector<double> draw()
        {
            auto t = engine->create_tensor({K_SIDE, K_SIDE});
            RandomOps random(recorder(), engine->rng());
            random.set_seed(K_SEED);
            GetParam().record(random, t);
            EXPECT_TRUE(recording->run_succeeds());
            return widened(read_tensor<float>(t));
        }
};

TEST_P(Small_InitializersTest, MomentsMatchTheInitializerLaw)
{
    expect_moments_match(draw(), GetParam().law);
}

TEST_P(Small_InitializersTest, DrawsStayInsideTheInitializerBound)
{
    const SampleMoments measured = measure_moments(draw());

    EXPECT_GE(measured.lowest, -GetParam().limit);
    EXPECT_LE(measured.highest, GetParam().limit);
}

INSTANTIATE_TEST_SUITE_P(
    , Small_InitializersTest, ::testing::ValuesIn(initializer_rows()),
    [](const ::testing::TestParamInfo<InitializerRow>& info) { return std::string(info.param.name); });
