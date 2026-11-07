/**************************************************************************/
/* test_normal.cpp                                                        */
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
 * @file test_normal.cpp
 * @brief Tests RandomOps::normal's sample moments, spread and seed determinism.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine_fixture/host_engine_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_readback.hpp>
#include <SushiBLAS/tensor_expectations/scalar_dtype.hpp>
#include "../random_statistics.hpp"
using namespace SushiBLAS::Test;

namespace
{
    /// @brief Holds the number of draws each statistical test takes.
    constexpr int64_t K_DRAWS = 100000;

    /// @brief Holds the requested mean.
    constexpr double K_MEAN = 2.0;

    /// @brief Holds the requested standard deviation.
    constexpr double K_STDDEV = 3.0;

    /// @brief Holds the distance from the mean, in standard deviations, that 100000 draws must reach on each side.
    constexpr double K_TAIL_REACH = 3.0;

    /// @brief Holds the seed the determinism tests share.
    constexpr uint64_t K_SEED = 20261006;

    /// @brief Holds a seed different from K_SEED.
    constexpr uint64_t K_OTHER_SEED = 7;
}

class Small_NormalTest : public HostEngineTest
{
    protected:
        /**
         * @brief Draws K_DRAWS normal samples of type T and returns them widened to double.
         * @tparam T float or double.
         */
        template <typename T>
        std::vector<double> draw(uint64_t seed)
        {
            auto t = engine->create_tensor({K_DRAWS}, sushiblas_dtype<T>());
            SushiBLAS::RandomOps(recorder(), engine->rng()).set_seed(seed);
            SushiBLAS::RandomOps(recorder(), engine->rng()).normal(t, K_MEAN, K_STDDEV);
            EXPECT_TRUE(recording->run_succeeds());
            return widened(read_tensor<T>(t));
        }

        /** @brief Draws K_DRAWS normal samples of the dtype from seed, widened to double. */
        std::vector<double> draw_as(SushiBLAS::Core::DataType dtype, uint64_t seed)
        {
            if (dtype == SushiBLAS::Core::DataType::Float64)
                return draw<double>(seed);
            return draw<float>(seed);
        }
};

/** @brief Runs one dtype of the moment test. */
class Small_NormalMoments : public Small_NormalTest,
                            public ::testing::WithParamInterface<SushiBLAS::Core::DataType>
{

};

TEST_P(Small_NormalMoments, MatchTheRequestedDistribution)
{
    const std::vector<double> samples = draw_as(GetParam(), K_SEED);

    expect_moments_match(samples, normal_law(K_MEAN, K_STDDEV));
}

TEST_P(Small_NormalMoments, SpreadReachesBothTails)
{
    const SampleMoments measured = measure_moments(draw_as(GetParam(), K_SEED));

    EXPECT_LT(measured.lowest, K_MEAN - K_TAIL_REACH * K_STDDEV);
    EXPECT_GT(measured.highest, K_MEAN + K_TAIL_REACH * K_STDDEV);
}

INSTANTIATE_TEST_SUITE_P(
    , Small_NormalMoments,
    ::testing::Values(SushiBLAS::Core::DataType::Float32, SushiBLAS::Core::DataType::Float64),
    [](const ::testing::TestParamInfo<SushiBLAS::Core::DataType>& info)
    { return std::string(SushiBLAS::Core::to_string(info.param)); });

TEST_F(Small_NormalTest, EqualSeedsDrawEqualSamples)
{
    const std::vector<double> first = draw<float>(K_SEED);
    const std::vector<double> second = draw<float>(K_SEED);
    EXPECT_EQ(first, second);
}

TEST_F(Small_NormalTest, DifferentSeedsDrawDifferentSamples)
{
    EXPECT_NE(draw<float>(K_SEED), draw<float>(K_OTHER_SEED));
}
