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
 * @file test_shuffle.cpp
 * @brief Tests that random().shuffle permutes its input and repeats for one seed and offset.
 * @author Mustafa Garip
 */

#include <vector>
#include <numeric>
#include <algorithm>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "../../common/test_utils.hpp"

using namespace SushiBLAS::Test;

class Unit_ShuffleTest : public SushiBLASTest
{
    protected:
        static constexpr int N = 64;

        std::vector<float> shuffled_iota(uint64_t seed)
        {
            std::vector<float> src(N);
            std::iota(src.begin(), src.end(), 0.0f);

            auto t = engine->create_tensor({N});
            fill_tensor(t, src);

            engine->random().set_seed(seed);
            engine->random().shuffle(t);
            engine->execute().wait();

            std::vector<float> out(N);
            auto& q = TestEnvironment::get_context().get_queue();
            q.memcpy(out.data(), t.data(), N * sizeof(float)).wait();
            return out;
        }
};

TEST_F(Unit_ShuffleTest, ResultIsAPermutationOfTheInput)
{
    auto out = shuffled_iota(1234);

    // The result is a permutation of the input. See tests/functional/unit/README.md.
    auto sorted = out;
    std::sort(sorted.begin(), sorted.end());
    for (int i = 0; i < N; ++i)
    {
        EXPECT_FLOAT_EQ(sorted[i], static_cast<float>(i)) << "Element " << i << " is not a permutation of the input.";
    }

    // The identity permutation means the kernel never ran. See tests/functional/unit/README.md.
    bool moved = false;
    for (int i = 0; i < N; ++i)
    {
        if (out[i] != static_cast<float>(i)) { moved = true; break; }
    }
    EXPECT_TRUE(moved) << "Shuffle left the tensor in its original order.";
}

TEST_F(Unit_ShuffleTest, SameSeedReproducesTheSamePermutation)
{
    auto first = shuffled_iota(20260803);

    // A fresh engine restarts the RNG offset counter. See tests/functional/unit/README.md.
    reinit_engine(SushiBLAS::Core::Layout::ROW_MAJOR);
    auto second = shuffled_iota(20260803);

    EXPECT_EQ(first, second);
}

TEST_F(Unit_ShuffleTest, DifferentSeedsProduceDifferentPermutations)
{
    auto first = shuffled_iota(1);

    reinit_engine(SushiBLAS::Core::Layout::ROW_MAJOR);
    auto second = shuffled_iota(2);

    EXPECT_NE(first, second) << "The seed is not reaching the permutation.";
}

TEST_F(Unit_ShuffleTest, ConsecutiveShufflesOnOneEngineDiffer)
{
    // Each task draws a fresh counter offset, so two shuffles must not repeat a permutation.
    std::vector<float> src(N);
    std::iota(src.begin(), src.end(), 0.0f);

    auto a = engine->create_tensor({N});
    auto b = engine->create_tensor({N});
    fill_tensor(a, src);
    fill_tensor(b, src);

    engine->random().set_seed(99);
    engine->random().shuffle(a);
    engine->random().shuffle(b);
    engine->execute().wait();

    std::vector<float> ha(N), hb(N);
    auto& q = TestEnvironment::get_context().get_queue();
    q.memcpy(ha.data(), a.data(), N * sizeof(float)).wait();
    q.memcpy(hb.data(), b.data(), N * sizeof(float)).wait();

    EXPECT_NE(ha, hb);
}

TEST_F(Unit_ShuffleTest, PermutesHalfPrecisionToo)
{
    // shuffle moves elements without computing on them, so it accepts HALF.
    constexpr int kSmall = 16;
    std::vector<float> src(kSmall);
    std::iota(src.begin(), src.end(), 0.0f);

    auto t = engine->create_tensor({kSmall}, SushiBLAS::Core::DataType::HALF);
    fill_tensor(t, src);

    engine->random().set_seed(5);
    engine->random().shuffle(t);
    engine->execute().wait();

    std::vector<sycl::half> out(kSmall);
    auto& q = TestEnvironment::get_context().get_queue();
    q.memcpy(out.data(), t.data(), kSmall * sizeof(sycl::half)).wait();

    std::vector<float> sorted;
    sorted.reserve(kSmall);
    for (auto h : out) sorted.push_back(static_cast<float>(h));
    std::sort(sorted.begin(), sorted.end());

    for (int i = 0; i < kSmall; ++i)
    {
        EXPECT_FLOAT_EQ(sorted[i], static_cast<float>(i));
    }
}
