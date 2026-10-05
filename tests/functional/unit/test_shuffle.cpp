/****************************************************************/
/* test_shuffle.cpp                                             */
/* SushiBLAS - https://github.com/SushiSystems/SushiBLAS        */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

// RandomOps::shuffle - previously declared in an installed header, implemented as a
// no-op stub, and commented out of the build, so any consumer that called it failed at
// link. Now a real Fisher-Yates permutation; these are the properties that make it one.

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

            engine->set_seed(seed);
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

    // The defining property, and the one the old stub trivially satisfied by doing
    // nothing - hence the second half of this test.
    auto sorted = out;
    std::sort(sorted.begin(), sorted.end());
    for (int i = 0; i < N; ++i)
    {
        EXPECT_FLOAT_EQ(sorted[i], static_cast<float>(i)) << "Element " << i << " is not a permutation of the input.";
    }

    // ...and it actually moved something. Identity is a legal permutation, but the odds
    // of drawing it from 64! are not distinguishable from zero, so seeing it means the
    // kernel never ran.
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

    // A fresh engine restarts the RNG offset counter, so this is the same (seed, offset)
    // pair the first run used - the reproducibility contract the header advertises.
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
    // Each task draws a fresh counter offset, so two shuffles in one graph must not
    // replay the same permutation - the failure mode a seed-only (offset-less) RNG has.
    std::vector<float> src(N);
    std::iota(src.begin(), src.end(), 0.0f);

    auto a = engine->create_tensor({N});
    auto b = engine->create_tensor({N});
    fill_tensor(a, src);
    fill_tensor(b, src);

    engine->set_seed(99);
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
    // shuffle moves elements rather than computing on them, so unlike the distributions
    // it has no reason to reject HALF.
    constexpr int kSmall = 16;
    std::vector<float> src(kSmall);
    std::iota(src.begin(), src.end(), 0.0f);

    auto t = engine->create_tensor({kSmall}, SushiBLAS::Core::DataType::HALF);
    fill_tensor(t, src);

    engine->set_seed(5);
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
