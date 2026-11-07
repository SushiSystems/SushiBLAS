/**************************************************************************/
/* test_random_large_lambda.cpp                                           */
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
 * @file test_random_large_lambda.cpp
 * @brief Tests the Poisson draw for lambda in the PTRS range and its reproducibility.
 * @author Mustafa Garip
 */

#include <cmath>
#include <vector>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine_fixture/host_engine_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_readback.hpp>
#include "../random_statistics.hpp"
using namespace SushiBLAS::Test;
using SushiBLAS::Core::DataType;

namespace
{
    /// @brief Number of draws per moment test, 2^16.
    constexpr int64_t K_DRAWS = 65536;
    /// @brief Seed of the moment tests.
    constexpr uint64_t K_MOMENT_SEED = 20261006;
    /// @brief Seed of the reproducibility tests.
    constexpr uint64_t K_REPLAY_SEED = 7;
    /// @brief Seed of the cap test.
    constexpr uint64_t K_CAP_SEED = 3;
    /// @brief Number of draws of the cap test.
    constexpr int64_t K_CAP_DRAWS = 4;

    /// @brief A mean in the PTRS range, near its lower end.
    constexpr double K_LAMBDA_SMALL = 50.0;
    /// @brief A mean in the middle of the PTRS range.
    constexpr double K_LAMBDA_MEDIUM = 5000.0;
    /// @brief A mean near the top of the accepted range.
    constexpr double K_LAMBDA_LARGE = 1.0e6;
    /// @brief A mean in the Knuth range, below the PTRS threshold of 10.
    constexpr double K_LAMBDA_KNUTH = 3.0;

    /** @brief Names one Poisson mean. */
    struct LambdaRow
    {
        const char* name; /**< The row name a failure shows. */
        double lambda;    /**< The mean of the draw. */
    };

    /** @brief Returns the means whose sample moments are checked. */
    std::vector<LambdaRow> moment_rows()
    {
        return {{"LambdaFifty", K_LAMBDA_SMALL},
                {"LambdaFiveThousand", K_LAMBDA_MEDIUM},
                {"LambdaOneMillion", K_LAMBDA_LARGE}};
    }

    /** @brief Returns the means whose draws are replayed: one per method. */
    std::vector<LambdaRow> replay_rows()
    {
        return {{"KnuthLambdaThree", K_LAMBDA_KNUTH}, {"PtrsLambdaFiveThousand", K_LAMBDA_MEDIUM}};
    }
}

class Small_RandomLargeLambdaTest : public HostEngineTest
{
    protected:
        /** @brief Returns count FLOAT64 Poisson draws of mean lambda from seed. */
        std::vector<double> draw(double lambda, uint64_t seed, int64_t count = K_DRAWS)
        {
            auto t = engine->create_tensor({count}, DataType::Float64);
            SushiBLAS::RandomOps(recorder(), engine->rng()).set_seed(seed);
            SushiBLAS::RandomOps(recorder(), engine->rng()).poisson(t, lambda);
            EXPECT_TRUE(recording->run_succeeds());
            return read_tensor<double>(t);
        }

        /** @brief Expects non-negative whole draws. */
        static void expect_counts(const std::vector<double>& values)
        {
            for (const double v : values)
            {
                ASSERT_GE(v, 0.0) << "A Poisson draw is negative.";
                ASSERT_EQ(v, std::floor(v)) << "A Poisson draw is not an integer.";
            }
        }
};

/** @brief Runs one row of moment_rows(). */
class Small_PoissonMoments : public Small_RandomLargeLambdaTest,
                             public ::testing::WithParamInterface<LambdaRow>
{

};

TEST_P(Small_PoissonMoments, MatchLambdaWithinStandardErrors)
{
    const double lambda = GetParam().lambda;
    const std::vector<double> values = draw(lambda, K_MOMENT_SEED);

    expect_counts(values);
    expect_moments_match(values, poisson_law(lambda));
}

INSTANTIATE_TEST_SUITE_P(
    , Small_PoissonMoments, ::testing::ValuesIn(moment_rows()),
    [](const ::testing::TestParamInfo<LambdaRow>& info) { return std::string(info.param.name); });

/** @brief Runs one row of replay_rows(). */
class Small_PoissonReplay : public Small_RandomLargeLambdaTest,
                            public ::testing::WithParamInterface<LambdaRow>
{

};

TEST_P(Small_PoissonReplay, ReplaysBitForBitFromOneSeed)
{
    const double lambda = GetParam().lambda;
    const std::vector<double> first = draw(lambda, K_REPLAY_SEED);
    const std::vector<double> second = draw(lambda, K_REPLAY_SEED);

    EXPECT_EQ(0, std::memcmp(first.data(), second.data(), first.size() * sizeof(double)));
}

INSTANTIATE_TEST_SUITE_P(
    , Small_PoissonReplay, ::testing::ValuesIn(replay_rows()),
    [](const ::testing::TestParamInfo<LambdaRow>& info) { return std::string(info.param.name); });

// A cap the record call refuses throws out of draw(), so this test also fails when the cap is not accepted.
TEST_F(Small_RandomLargeLambdaTest, PoissonAtTheCapDrawsNonNegativeIntegers)
{
    expect_counts(draw(SushiBLAS::K_MAX_POISSON_LAMBDA, K_CAP_SEED, K_CAP_DRAWS));
}
