/**************************************************************************/
/* test_shuffle.cpp                                                       */
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
 * @file test_shuffle.cpp
 * @brief Tests that RandomOps::shuffle permutes its input and repeats for one seed and offset.
 * @author Mustafa Garip
 */

#include <vector>
#include <numeric>
#include <algorithm>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_fill.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
using namespace SushiBLAS::Test;

class Medium_ShuffleTest : public SushiBLASTest
{
    protected:
        static constexpr int K_N = 64;

        std::vector<float> shuffled_iota(uint64_t seed)
        {
            std::vector<float> src(K_N);
            std::iota(src.begin(), src.end(), 0.0f);

            auto t = engine->create_tensor({K_N});
            fill_tensor(t, src);

            SushiBLAS::RandomOps(recorder(), engine->rng()).set_seed(seed);
            SushiBLAS::RandomOps(recorder(), engine->rng()).shuffle(t);
            recording->run().value();

            std::vector<float> out(K_N);
            auto& q = TestEnvironment::queue();
            q.memcpy(out.data(), t.data(), K_N * sizeof(float)).wait();
            return out;
        }

        /** @brief Expects the values to be 0, 1, 2, ... in some order. */
        static void expect_iota_set(std::vector<float> values)
        {
            std::sort(values.begin(), values.end());

            std::vector<float> iota(values.size());
            std::iota(iota.begin(), iota.end(), 0.0f);
            EXPECT_EQ(values, iota);
        }
};

TEST_F(Medium_ShuffleTest, ResultIsAPermutationOfTheInput)
{
    // The result is a permutation of the input. See tests/operations/random/README.md.
    expect_iota_set(shuffled_iota(1234));
}

TEST_F(Medium_ShuffleTest, ResultIsNotTheIdentity)
{
    // The identity permutation means the kernel never ran. See tests/operations/random/README.md.
    std::vector<float> iota(K_N);
    std::iota(iota.begin(), iota.end(), 0.0f);

    EXPECT_NE(shuffled_iota(1234), iota) << "Shuffle left the tensor in its original order.";
}

TEST_F(Medium_ShuffleTest, SameSeedReproducesTheSamePermutation)
{
    auto first = shuffled_iota(20260803);

    // A fresh engine restarts the RNG offset counter. See tests/operations/random/README.md.
    reinit_engine(SushiBLAS::Core::Layout::RowMajor);
    auto second = shuffled_iota(20260803);

    EXPECT_EQ(first, second);
}

TEST_F(Medium_ShuffleTest, DifferentSeedsProduceDifferentPermutations)
{
    auto first = shuffled_iota(1);

    reinit_engine(SushiBLAS::Core::Layout::RowMajor);
    auto second = shuffled_iota(2);

    EXPECT_NE(first, second) << "The seed is not reaching the permutation.";
}

TEST_F(Medium_ShuffleTest, ConsecutiveShufflesOnOneEngineDiffer)
{
    // Each task draws a fresh counter offset, so two shuffles must not repeat a permutation.
    std::vector<float> src(K_N);
    std::iota(src.begin(), src.end(), 0.0f);

    auto a = engine->create_tensor({K_N});
    auto b = engine->create_tensor({K_N});
    fill_tensor(a, src);
    fill_tensor(b, src);

    SushiBLAS::RandomOps(recorder(), engine->rng()).set_seed(99);
    SushiBLAS::RandomOps(recorder(), engine->rng()).shuffle(a);
    SushiBLAS::RandomOps(recorder(), engine->rng()).shuffle(b);
    recording->run().value();

    std::vector<float> ha(K_N), hb(K_N);
    auto& q = TestEnvironment::queue();
    q.memcpy(ha.data(), a.data(), K_N * sizeof(float)).wait();
    q.memcpy(hb.data(), b.data(), K_N * sizeof(float)).wait();

    EXPECT_NE(ha, hb);
}

TEST_F(Medium_ShuffleTest, PermutesHalfPrecisionToo)
{
    // shuffle moves elements without computing on them, so it accepts HALF.
    constexpr int kSmall = 16;
    std::vector<float> src(kSmall);
    std::iota(src.begin(), src.end(), 0.0f);

    auto t = engine->create_tensor({kSmall}, SushiBLAS::Core::DataType::Half);
    fill_tensor(t, src);

    SushiBLAS::RandomOps(recorder(), engine->rng()).set_seed(5);
    SushiBLAS::RandomOps(recorder(), engine->rng()).shuffle(t);
    recording->run().value();

    std::vector<sycl::half> out(kSmall);
    auto& q = TestEnvironment::queue();
    q.memcpy(out.data(), t.data(), kSmall * sizeof(sycl::half)).wait();

    expect_iota_set(std::vector<float>(out.begin(), out.end()));
}
