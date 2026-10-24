/**************************************************************************/
/* test_rng_stream.cpp                                                    */
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
 * @file test_rng_stream.cpp
 * @brief Tests RNGStream's reseed, offset reservation, thread safety and seed determinism.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <cstring>
#include <functional>
#include <set>
#include <thread>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine/rng_stream.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_readback.hpp>
using namespace SushiBLAS::Test;

namespace
{
    /// @brief Holds the seed the stream tests share.
    constexpr uint64_t K_SEED = 20261007;

    /// @brief Holds the number of offsets each thread reserves.
    constexpr int K_RESERVATIONS_PER_THREAD = 10000;

    /// @brief Holds the element count of the normal fill.
    constexpr int64_t K_FILL_SIZE = 4096;

    /// @brief Reserves K_RESERVATIONS_PER_THREAD single offsets from @p stream into @p out.
    void reserve_many(SushiBLAS::RNGStream& stream, std::vector<uint64_t>& out)
    {
        out.reserve(K_RESERVATIONS_PER_THREAD);
        for (int i = 0; i < K_RESERVATIONS_PER_THREAD; ++i)
            out.push_back(stream.reserve(1).offset);
    }
}

class Medium_RngStreamTest : public SushiBLASTest
{
    protected:
        /// @brief Seeds the engine, fills a FLOAT32 tensor with normal samples and returns them.
        std::vector<float> normal_fill(uint64_t seed)
        {
            auto t = engine->create_tensor({K_FILL_SIZE});
            SushiBLAS::RandomOps(recorder(), engine->rng()).set_seed(seed);
            SushiBLAS::RandomOps(recorder(), engine->rng()).normal(t, 0.0, 1.0);
            recording->run().value();
            return read_tensor<float>(t);
        }
};

TEST_F(Medium_RngStreamTest, ReseedRewindsTheOffset)
{
    SushiBLAS::RNGStream stream;
    stream.reseed(K_SEED);
    (void)stream.reserve(5);
    (void)stream.reserve(3);

    stream.reseed(K_SEED + 1);

    const SushiBLAS::RNGPosition position = stream.reserve(1);
    EXPECT_EQ(stream.seed(), K_SEED + 1);
    EXPECT_EQ(position.seed, K_SEED + 1);
    EXPECT_EQ(position.offset, 0u);
}

TEST_F(Medium_RngStreamTest, ReserveReturnsConsecutiveDisjointRanges)
{
    SushiBLAS::RNGStream stream;
    stream.reseed(K_SEED);

    EXPECT_EQ(stream.reserve(4).offset, 0u);
    EXPECT_EQ(stream.reserve(1).offset, 4u);
    EXPECT_EQ(stream.reserve(10).offset, 5u);
    EXPECT_EQ(stream.reserve(1).offset, 15u);
}

TEST_F(Medium_RngStreamTest, ConcurrentReservationsAreDistinct)
{
    SushiBLAS::RNGStream stream;
    stream.reseed(K_SEED);

    std::vector<uint64_t> first;
    std::vector<uint64_t> second;
    std::thread a(reserve_many, std::ref(stream), std::ref(first));
    std::thread b(reserve_many, std::ref(stream), std::ref(second));
    a.join();
    b.join();

    std::set<uint64_t> merged(first.begin(), first.end());
    merged.insert(second.begin(), second.end());
    EXPECT_EQ(merged.size(), static_cast<size_t>(2 * K_RESERVATIONS_PER_THREAD));
}

TEST_F(Medium_RngStreamTest, ReseedRacingReservationsNeverPairsAForeignSeed)
{
    SushiBLAS::RNGStream stream;
    stream.reseed(K_SEED);

    std::vector<SushiBLAS::RNGPosition> positions;
    positions.reserve(K_RESERVATIONS_PER_THREAD);
    std::thread reserver([&stream, &positions]()
    {
        for (int i = 0; i < K_RESERVATIONS_PER_THREAD; ++i)
            positions.push_back(stream.reserve(1));
    });
    std::thread reseeder([&stream]()
    {
        for (int i = 0; i < K_RESERVATIONS_PER_THREAD; ++i)
            stream.reseed(i % 2 == 0 ? K_SEED + 1 : K_SEED);
    });
    reserver.join();
    reseeder.join();

    ASSERT_EQ(positions.size(), static_cast<size_t>(K_RESERVATIONS_PER_THREAD));
    for (const SushiBLAS::RNGPosition& position : positions)
        EXPECT_TRUE(position.seed == K_SEED || position.seed == K_SEED + 1)
            << "Seed " << position.seed << " is neither of the two seeds used.";
}

TEST_F(Medium_RngStreamTest, EqualSeedsGiveBitEqualNormalFills)
{
    const std::vector<float> first = normal_fill(K_SEED);
    const std::vector<float> second = normal_fill(K_SEED);

    ASSERT_EQ(first.size(), second.size());
    EXPECT_EQ(std::memcmp(first.data(), second.data(), first.size() * sizeof(float)), 0);
}
