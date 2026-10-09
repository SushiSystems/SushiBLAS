/**************************************************************************/
/* test_reduce_segments.cpp                                               */
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
 * @file test_reduce_segments.cpp
 * @brief Tests Reduce::reduce_segments in both launch shapes and the axis fold that runs on it.
 * @author Mustafa Garip
 */

#include <array>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/SushiBLAS.h>
#include <SushiBLAS/support/reduce_1d.hpp>
#include <SushiBLAS/support/reduce_axis.hpp>
#include "test_utils.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief Holds the lane counts of the two launch shapes: one item per segment, and a work-group of eight.
            constexpr std::size_t K_LANE_COUNTS[] = {1, 8};

            /// @brief Holds the segment count of the ramp cases.
            constexpr std::size_t K_RAMP_SEGMENTS = 5;

            /// @brief Holds the gap between the first terms of consecutive ramp segments.
            constexpr std::size_t K_RAMP_SCALE = 1000;

            /// @brief Holds the extents the ramp is folded over; they straddle the eight-lane work-group.
            constexpr std::size_t K_RAMP_EXTENTS[] = {0, 1, 7, 8, 9, 1000};

            /// @brief Holds the extent the counting store is run over.
            constexpr std::size_t K_COUNTED_EXTENT = 100;

            /// @brief Holds a value no ramp case stores.
            constexpr std::int64_t K_UNWRITTEN = -1;

            /// @brief Holds the lane count of the order-sensitive case.
            constexpr std::size_t K_PATTERN_LANES = 4;

            /// @brief Holds the segment count of the order-sensitive case; segment s starts s terms into the pattern.
            constexpr std::size_t K_PATTERN_SEGMENTS = 3;

            /// @brief Holds an extent that leaves the last of the four lanes one term short.
            constexpr std::size_t K_PATTERN_EXTENT = 103;

            /// @brief Holds the segments per group of the axis cases.
            constexpr std::size_t K_AXIS_SEGMENTS = 3;

            /// @brief Holds an extent of the axis cases that no lane count of this file divides.
            constexpr std::size_t K_AXIS_EXTENT = 37;

            /// @brief Holds the group counts of the axis cases: one segment axis, and two.
            constexpr std::size_t K_AXIS_GROUP_COUNTS[] = {1, 3};

            /// @brief Holds a HALF value whose double is an infinity in HALF.
            constexpr float K_HALF_LARGE = 60000.0f;

            /// @brief Holds the height and the width of the HALF map.
            constexpr int64_t K_MAP_SIDE = 8;

            /// @brief Holds the channel count of the HALF map.
            constexpr int64_t K_MAP_CHANNELS = 2;

            /// @brief Holds the value of every element of the HALF map; 64 of them sum to 128 000.
            constexpr float K_MAP_VALUE = 2000.0f;

            /** @brief Names a host vector whose elements live in shared USM, where a kernel reaches them. */
            template<typename T>
            using SharedVector = std::vector<T, sycl::usm_allocator<T, sycl::usm::alloc::shared>>;

            /** @brief Returns count value-initialised elements of shared USM in the context of q. */
            template<typename T>
            [[nodiscard]] SharedVector<T> shared_vector(sycl::queue& q, std::size_t count)
            {
                return SharedVector<T>(count,
                                       sycl::usm_allocator<T, sycl::usm::alloc::shared>(q));
            }

            /** @brief Returns whether a and b hold the same bits. */
            template<typename T>
            [[nodiscard]] bool same_bits(const T& a, const T& b) noexcept
            {
                return std::memcmp(&a, &b, sizeof(T)) == 0;
            }

            /** @brief Returns the name a failure message gives the launch shape of a lane count. */
            [[nodiscard]] const char* shape_name(std::size_t lanes) noexcept
            {
                return lanes == 1 ? "item-per-segment shape" : "work-group tree";
            }

            /** @brief Returns term k of a segment's ramp: `segment * K_RAMP_SCALE + k`. */
            [[nodiscard]] constexpr std::int64_t ramp_term(std::size_t segment,
                                                           std::size_t k) noexcept
            {
                return static_cast<std::int64_t>(segment * K_RAMP_SCALE + k);
            }

            /** @brief Returns the sum of the first extent ramp terms of a segment. */
            [[nodiscard]] constexpr std::int64_t ramp_sum(std::size_t segment,
                                                          std::size_t extent) noexcept
            {
                const std::int64_t n = static_cast<std::int64_t>(extent);
                return n * ramp_term(segment, 0) + n * (n - 1) / 2;
            }

            /** @brief Returns term k of the pattern `1e8, 1, -1e8, 1`, whose FLOAT32 sum depends on the order. */
            [[nodiscard]] constexpr float order_sensitive_term(std::size_t k) noexcept
            {
                switch (k % 4)
                {
                    case 0: return 1e8f;
                    case 2: return -1e8f;
                    default: return 1.0f;
                }
            }

            /** @brief Returns element i of an input whose neighbours lie six orders of magnitude apart. */
            template<typename T>
            [[nodiscard]] T mixed_magnitude(std::size_t i) noexcept
            {
                const T base = T(0.1) + static_cast<T>((i * 37) % 23) * T(0.031);
                return i % 5 == 0 ? base * T(1e6) : base;
            }

            /** @brief Reads the ramp terms of one segment. */
            class RampCursor final
            {
                public:
                    /** @brief Binds the cursor to a segment. */
                    explicit RampCursor(std::size_t segment) noexcept
                        : segment_(segment)
                    {

                    }

                    /** @brief Returns term k of the segment's ramp. */
                    [[nodiscard]] std::int64_t operator()(std::size_t k) const noexcept
                    {
                        return ramp_term(segment_, k);
                    }

                private:
                    /// @brief The segment the cursor reads.
                    std::size_t segment_;
            };

            /** @brief Opens the ramp of each segment for Reduce::reduce_segments. */
            class RampOpen final
            {
                public:
                    /** @brief Returns the cursor of a segment. */
                    [[nodiscard]] RampCursor operator()(std::size_t segment) const noexcept
                    {
                        return RampCursor(segment);
                    }
            };

            /** @brief Reads the order-sensitive pattern from a starting term. */
            class PatternCursor final
            {
                public:
                    /** @brief Binds the cursor to the pattern term its term 0 is. */
                    explicit PatternCursor(std::size_t start) noexcept
                        : start_(start)
                    {

                    }

                    /** @brief Returns pattern term `start + k`. */
                    [[nodiscard]] float operator()(std::size_t k) const noexcept
                    {
                        return order_sensitive_term(start_ + k);
                    }

                private:
                    /// @brief The pattern term the cursor's term 0 is.
                    std::size_t start_;
            };

            /** @brief Opens segment s of the order-sensitive case s terms into the pattern. */
            class PatternOpen final
            {
                public:
                    /** @brief Returns the cursor of a segment. */
                    [[nodiscard]] PatternCursor operator()(std::size_t segment) const noexcept
                    {
                        return PatternCursor(segment);
                    }
            };

            /** @brief Stores each segment's value at the segment's own index of an array. */
            template<typename W>
            class IndexedStore final
            {
                public:
                    /** @brief Binds the store to an array of one element per segment. */
                    explicit IndexedStore(W* out) noexcept
                        : out_(out)
                    {

                    }

                    /** @brief Writes value to the element of a segment. */
                    void operator()(std::size_t segment, W value) const
                    {
                        out_[segment] = value;
                    }

                private:
                    /// @brief The array's first element.
                    W* out_;
            };

            /** @brief Holds what the counting store saw for one segment index. */
            struct StoreRecord
            {
                std::int32_t calls = 0;    /**< How many calls carried this index. */
                std::int64_t segment = -1; /**< The index the last call carried. */
                std::int64_t value = 0;    /**< The value the last call carried. */
            };

            /** @brief Counts the calls each segment index receives and keeps the index beside the value. */
            class CountingStore final
            {
                public:
                    /**
                     * @brief Binds the store to its records.
                     * @param records One record per segment, then one for every index past the last segment.
                     */
                    CountingStore(StoreRecord* records, std::size_t segments) noexcept
                        : records_(records), segments_(segments)
                    {

                    }

                    /** @brief Records one call; the count is atomic, so calls from two work-items both show. */
                    void operator()(std::size_t segment, std::int64_t value) const
                    {
                        StoreRecord& record = records_[segment < segments_ ? segment : segments_];

                        sycl::atomic_ref<std::int32_t, sycl::memory_order::relaxed,
                                         sycl::memory_scope::device>(record.calls).fetch_add(1);
                        record.segment = static_cast<std::int64_t>(segment);
                        record.value = value;
                    }

                private:
                    /// @brief The first record.
                    StoreRecord* records_;

                    /// @brief How many segments the fold has.
                    std::size_t segments_;
            };

            /**
             * @brief Folds `term(0)` to `term(extent - 1)` on the host in the order reduce_segments folds one segment.
             * @param lanes The work-items of the launch shape; a power of two, 1 for the item-per-segment shape.
             * @param term Callable `(std::size_t k) -> W`.
             */
            template<template<typename> class Accum, typename W, typename Term>
            [[nodiscard]] W fold_in_lane_order(std::size_t extent, std::size_t lanes,
                                               const Term& term)
            {
                std::vector<W> tile(lanes);
                for (std::size_t lane = 0; lane < lanes; ++lane)
                {
                    Accum<W> acc;
                    for (std::size_t k = lane; k < extent; k += lanes)
                        acc.accumulate(term(k));

                    tile[lane] = acc.value();
                }

                for (std::size_t off = lanes / 2; off > 0; off >>= 1)
                {
                    for (std::size_t lane = 0; lane < off; ++lane)
                        tile[lane] = Accum<W>::combine(tile[lane], tile[lane + off]);
                }

                return tile[0];
            }

            /**
             * @brief Expects reduce_axis to write, for every segment of an `[groups, extent, segments]` array, the bits of the host fold in lane order.
             * @tparam T float or double.
             */
            template<typename T>
            void expect_axis_sums_in_lane_order(sycl::queue& q, std::size_t groups,
                                                std::size_t lanes)
            {
                SCOPED_TRACE(shape_name(lanes));
                SCOPED_TRACE(groups == 1 ? "one segment axis" : "two segment axes");

                Reduce::AxisSpan span;
                span.segments = K_AXIS_SEGMENTS;
                span.segments2 = groups;
                span.extent = K_AXIS_EXTENT;
                span.segment_stride = 1;
                span.element_stride = K_AXIS_SEGMENTS;
                span.segment_stride2 = K_AXIS_EXTENT * K_AXIS_SEGMENTS;
                span.out_stride = 1;

                SharedVector<T> in = shared_vector<T>(q, groups * span.segment_stride2);
                for (std::size_t i = 0; i < in.size(); ++i)
                    in[i] = mixed_magnitude<T>(i);

                SharedVector<T> out = shared_vector<T>(q, span.total_segments());
                const T* source = in.data();

                Reduce::reduce_axis<Reduce::CompensatedSum>(
                    q, source, out.data(), span, Reduce::AxisPlan{lanes},
                    [](T x) { return x; },
                    [](T value, std::size_t) { return value; },
                    {}).wait();

                for (std::size_t s = 0; s < span.total_segments(); ++s)
                {
                    const T* base = source + (s / span.segments) * span.segment_stride2
                                           + (s % span.segments) * span.segment_stride;
                    const std::size_t element_stride = span.element_stride;

                    const T want = fold_in_lane_order<Reduce::CompensatedSum, T>(
                        span.extent, lanes,
                        [base, element_stride](std::size_t k) { return base[k * element_stride]; });

                    EXPECT_TRUE(same_bits(out[s], want))
                        << "segment " << s << " is " << static_cast<double>(out[s])
                        << ", the host fold in lane order gives " << static_cast<double>(want);
                }
            }
        } // namespace

        /** @brief Runs the segmented fold on the test queue and on the fixture's engine. */
        class Unit_ReduceSegments : public SushiBLASTest
        {
            protected:
                /** @brief Returns the queue every case submits to. */
                [[nodiscard]] static sycl::queue& queue()
                {
                    return TestEnvironment::get_context().get_queue();
                }
        };

        /** @brief Tests that both launch shapes fold every term of every segment once, for extents around the lane count and for none. */
        TEST_F(Unit_ReduceSegments, BothShapesFoldEverySegmentOnce)
        {
            sycl::queue& q = queue();
            SharedVector<std::int64_t> sums = shared_vector<std::int64_t>(q, K_RAMP_SEGMENTS);

            for (const std::size_t lanes : K_LANE_COUNTS)
            {
                SCOPED_TRACE(shape_name(lanes));

                for (const std::size_t extent : K_RAMP_EXTENTS)
                {
                    std::fill(sums.begin(), sums.end(), K_UNWRITTEN);

                    Reduce::reduce_segments<Reduce::ExactSum, std::int64_t>(
                        q, K_RAMP_SEGMENTS, extent, Reduce::AxisPlan{lanes}, RampOpen{},
                        IndexedStore<std::int64_t>(sums.data()), {}).wait();

                    for (std::size_t s = 0; s < K_RAMP_SEGMENTS; ++s)
                    {
                        EXPECT_EQ(sums[s], ramp_sum(s, extent))
                            << "segment " << s << " over " << extent << " terms";
                    }
                }
            }
        }

        /**
         * @brief Tests that the work-group shape folds lane `k % 4` in ascending k and joins the lanes as `(0 + 2) + (1 + 3)`.
         * @see tests/unit/README.md
         */
        TEST_F(Unit_ReduceSegments, TheCooperativeShapeFoldsInLaneOrder)
        {
            using Sum = Reduce::CompensatedSum<float>;

            sycl::queue& q = queue();
            SharedVector<float> sums = shared_vector<float>(q, K_PATTERN_SEGMENTS);

            Reduce::reduce_segments<Reduce::CompensatedSum, float>(
                q, K_PATTERN_SEGMENTS, K_PATTERN_EXTENT, Reduce::AxisPlan{K_PATTERN_LANES},
                PatternOpen{}, IndexedStore<float>(sums.data()), {}).wait();

            for (std::size_t s = 0; s < K_PATTERN_SEGMENTS; ++s)
            {
                std::array<Sum, K_PATTERN_LANES> lanes{};
                for (std::size_t k = 0; k < K_PATTERN_EXTENT; ++k)
                    lanes[k % K_PATTERN_LANES].accumulate(order_sensitive_term(s + k));

                const float strided = Sum::combine(
                    Sum::combine(lanes[0].value(), lanes[2].value()),
                    Sum::combine(lanes[1].value(), lanes[3].value()));
                const float adjacent = Sum::combine(
                    Sum::combine(lanes[0].value(), lanes[1].value()),
                    Sum::combine(lanes[2].value(), lanes[3].value()));

                // A pattern both trees sum alike proves no order; see tests/unit/README.md.
                ASSERT_FALSE(same_bits(strided, adjacent))
                    << "segment " << s << ": both trees give " << strided;

                EXPECT_TRUE(same_bits(sums[s], strided))
                    << "segment " << s << " is " << sums[s] << ", the lane order gives "
                    << strided;
            }
        }

        /** @brief Tests that reduce_axis writes the bits of a host fold in lane order for FLOAT32 and FLOAT64, in both shapes, with one segment axis and with two. */
        TEST_F(Unit_ReduceSegments, ReduceAxisKeepsItsFloatBits)
        {
            sycl::queue& q = queue();

            for (const std::size_t groups : K_AXIS_GROUP_COUNTS)
            {
                for (const std::size_t lanes : K_LANE_COUNTS)
                    expect_axis_sums_in_lane_order<float>(q, groups, lanes);
            }

            if (!q.get_device().has(sycl::aspect::fp64))
                GTEST_SKIP() << "FLOAT32 was compared; the device does not report the fp64 "
                                "aspect, so FLOAT64 was not";

            for (const std::size_t groups : K_AXIS_GROUP_COUNTS)
            {
                for (const std::size_t lanes : K_LANE_COUNTS)
                    expect_axis_sums_in_lane_order<double>(q, groups, lanes);
            }
        }

        /**
         * @brief Tests that a HALF segment holding `60000, 60000, -60000` in one lane reduces to 60000 in both shapes.
         * @see tests/unit/README.md
         */
        TEST_F(Unit_ReduceSegments, AHalfAxisSumIsFoldedInFloat)
        {
            sycl::queue& q = queue();
            if (!q.get_device().has(sycl::aspect::fp16))
                GTEST_SKIP() << "the device does not report the fp16 aspect";

            const sycl::half want(K_HALF_LARGE);

            for (const std::size_t lanes : K_LANE_COUNTS)
            {
                SCOPED_TRACE(shape_name(lanes));

                // The three values sit lanes elements apart, so lane 0 folds all three in order.
                const std::size_t extent = 2 * lanes + 1;
                SharedVector<sycl::half> row = shared_vector<sycl::half>(q, extent);
                row[0] = sycl::half(K_HALF_LARGE);
                row[lanes] = sycl::half(K_HALF_LARGE);
                row[2 * lanes] = sycl::half(-K_HALF_LARGE);

                SharedVector<sycl::half> sum = shared_vector<sycl::half>(q, 1);

                Reduce::AxisSpan span;
                span.segments = 1;
                span.extent = extent;
                span.segment_stride = extent;

                const sycl::half* in = row.data();
                Reduce::reduce_axis<Reduce::CompensatedSum>(
                    q, in, sum.data(), span, Reduce::AxisPlan{lanes},
                    [](sycl::half x) { return x; },
                    [](float value, std::size_t) { return value; },
                    {}).wait();

                EXPECT_TRUE(same_bits(sum[0], want))
                    << "the sum is " << static_cast<float>(sum[0])
                    << "; a fold in HALF passes through an infinity";
            }
        }

        /** @brief Tests that mean_spatial of an 8 by 8 HALF map of 2000 is 2000 in each channel under ForceSycl, though each channel sums to 128 000. */
        TEST_F(Unit_ReduceSegments, AHalfMeanSpatialDoesNotOverflow)
        {
            if (!queue().get_device().has(sycl::aspect::fp16))
                GTEST_SKIP() << "the device does not report the fp16 aspect";

            engine->set_mode(ExecutionMode::ForceSycl);

            Tensor map = engine->create_tensor({1, K_MAP_SIDE, K_MAP_SIDE, K_MAP_CHANNELS},
                                               Core::DataType::Half);
            Tensor means = engine->create_tensor({1, K_MAP_CHANNELS}, Core::DataType::Half);
            fill_tensor(map, std::vector<float>(static_cast<std::size_t>(map.num_elements),
                                                K_MAP_VALUE));

            SushiBLAS::ReductionOps(*engine).mean_spatial(map, means);
            engine->execute();

            const std::vector<sycl::half> got = read_tensor<sycl::half>(means);
            ASSERT_EQ(got.size(), static_cast<std::size_t>(K_MAP_CHANNELS));

            const sycl::half want(K_MAP_VALUE);
            for (std::size_t c = 0; c < got.size(); ++c)
            {
                EXPECT_TRUE(same_bits(got[c], want))
                    << "the mean of channel " << c << " is " << static_cast<float>(got[c]);
            }
        }

        /** @brief Tests that both launch shapes call the store once for each segment index, with that segment's value, and for no other index. */
        TEST_F(Unit_ReduceSegments, AStoreSeesItsOwnSegmentOnly)
        {
            sycl::queue& q = queue();
            SharedVector<StoreRecord> records = shared_vector<StoreRecord>(q, K_RAMP_SEGMENTS + 1);

            for (const std::size_t lanes : K_LANE_COUNTS)
            {
                SCOPED_TRACE(shape_name(lanes));
                std::fill(records.begin(), records.end(), StoreRecord{});

                Reduce::reduce_segments<Reduce::ExactSum, std::int64_t>(
                    q, K_RAMP_SEGMENTS, K_COUNTED_EXTENT, Reduce::AxisPlan{lanes}, RampOpen{},
                    CountingStore(records.data(), K_RAMP_SEGMENTS), {}).wait();

                for (std::size_t s = 0; s < K_RAMP_SEGMENTS; ++s)
                {
                    EXPECT_EQ(records[s].calls, 1) << "segment " << s;
                    EXPECT_EQ(records[s].segment, static_cast<std::int64_t>(s));
                    EXPECT_EQ(records[s].value, ramp_sum(s, K_COUNTED_EXTENT)) << "segment " << s;
                }

                EXPECT_EQ(records[K_RAMP_SEGMENTS].calls, 0)
                    << "the store was called with an index past the last segment";
            }
        }

    } // namespace Test
} // namespace SushiBLAS
