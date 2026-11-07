/**************************************************************************/
/* test_uniform.cpp                                                       */
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
 * @file test_uniform.cpp
 * @brief Tests RandomOps::uniform's bounds, sample moments, spread and seed determinism.
 * @author Mustafa Garip
 */

#include <cmath>
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

    /// @brief Holds the lower bound of the range.
    constexpr double K_MIN = -2.0;

    /// @brief Holds the upper bound of the range.
    constexpr double K_MAX = 6.0;

    /// @brief Holds the chance, per edge, that no draw falls in the edge band: (1 - band / width)^draws equals it.
    constexpr double K_EDGE_FALSE_FAILURE = 1.0e-9;

    /// @brief Holds the seed the determinism tests share.
    constexpr uint64_t K_SEED = 20261006;

    /// @brief Holds a seed different from K_SEED.
    constexpr uint64_t K_OTHER_SEED = 7;

    /** @brief Returns the band at each end the draws must reach: width * ln(1 / false failure) / draws. */
    double edge_band()
    {
        return (K_MAX - K_MIN) * std::log(1.0 / K_EDGE_FALSE_FAILURE) / static_cast<double>(K_DRAWS);
    }
}

class Small_UniformTest : public HostEngineTest
{
    protected:
        /**
         * @brief Draws K_DRAWS uniform samples of type T and returns them widened to double.
         * @tparam T float or double.
         */
        template <typename T>
        std::vector<double> draw(uint64_t seed)
        {
            auto t = engine->create_tensor({K_DRAWS}, sushiblas_dtype<T>());
            SushiBLAS::RandomOps(recorder(), engine->rng()).set_seed(seed);
            SushiBLAS::RandomOps(recorder(), engine->rng()).uniform(t, K_MIN, K_MAX);
            EXPECT_TRUE(recording->run_succeeds());
            return widened(read_tensor<T>(t));
        }

        /** @brief Draws K_DRAWS uniform samples of the dtype from seed, widened to double. */
        std::vector<double> draw_as(SushiBLAS::Core::DataType dtype, uint64_t seed)
        {
            if (dtype == SushiBLAS::Core::DataType::Float64)
                return draw<double>(seed);
            return draw<float>(seed);
        }
};

/** @brief Runs one dtype of the range and moment tests. */
class Small_UniformDraw : public Small_UniformTest,
                          public ::testing::WithParamInterface<SushiBLAS::Core::DataType>
{

};

TEST_P(Small_UniformDraw, StaysInTheHalfOpenRange)
{
    const SampleMoments measured = measure_moments(draw_as(GetParam(), K_SEED));

    EXPECT_GE(measured.lowest, K_MIN);
    EXPECT_LT(measured.highest, K_MAX);
}

TEST_P(Small_UniformDraw, FillsTheRangeToBothEdges)
{
    const SampleMoments measured = measure_moments(draw_as(GetParam(), K_SEED));

    EXPECT_LT(measured.lowest, K_MIN + edge_band());
    EXPECT_GT(measured.highest, K_MAX - edge_band());
}

TEST_P(Small_UniformDraw, MomentsMatchTheUniformLaw)
{
    expect_moments_match(draw_as(GetParam(), K_SEED), uniform_law(K_MIN, K_MAX));
}

INSTANTIATE_TEST_SUITE_P(
    , Small_UniformDraw,
    ::testing::Values(SushiBLAS::Core::DataType::Float32, SushiBLAS::Core::DataType::Float64),
    [](const ::testing::TestParamInfo<SushiBLAS::Core::DataType>& info)
    { return std::string(SushiBLAS::Core::to_string(info.param)); });

TEST_F(Small_UniformTest, EqualSeedsDrawEqualSamples)
{
    const std::vector<double> first = draw<float>(K_SEED);
    const std::vector<double> second = draw<float>(K_SEED);
    EXPECT_EQ(first, second);
}

TEST_F(Small_UniformTest, DifferentSeedsDrawDifferentSamples)
{
    EXPECT_NE(draw<float>(K_SEED), draw<float>(K_OTHER_SEED));
}
