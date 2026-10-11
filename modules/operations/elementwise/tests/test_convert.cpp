/**************************************************************************/
/* test_convert.cpp                                                       */
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
 * @file test_convert.cpp
 * @brief Tests ElementwiseOps::convert between the real dtypes, its refusals and its ordering.
 * @author Mustafa Garip
 */

#include <cmath>
#include <limits>
#include <string>
#include <vector>
#include <cstddef>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include "test_utils.hpp"
#include "graph_probe.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief The three dtypes `convert` serves, in the order the switch lists them. */
            const std::vector<Core::DataType> K_REAL_DTYPES =
                {Core::DataType::Half, Core::DataType::Float32, Core::DataType::Float64};

            /**
             * @brief The nearest IEEE-754 binary16 value to 0.1, as an exact double.
             *
             * 0.1 = 1.100110011001100110... x 2^-4. Keeping ten mantissa bits gives
             * 1.1001100110 (the first dropped bit is 0, so it rounds down), which is
             * 1.599609375 x 2^-4. Not equal to 0.1f, which is the point.
             */
            constexpr double K_TENTH_AS_HALF = 0.0999755859375;

            /** @brief The largest finite binary16 value. */
            constexpr double K_HALF_MAX = 65504.0;

            /**
             * @brief 2 x 2^-24: the binary16 subnormal nearest to 1e-7.
             *
             * 1e-7 / 2^-24 is 1.677..., which rounds to 2, so the conversion lands on a
             * subnormal rather than flushing to zero.
             */
            constexpr double K_TINY_AS_HALF = 1.1920928955078125e-7;
        }

        class Unit_ConvertTest : public SushiBLASTest
        {
            protected:
                /**
                 * @brief Reads a whole real tensor back to the host, widened to double.
                 *
                 * Widening HALF or FLOAT32 to double is exact, so an `EXPECT_EQ` against a
                 * double literal below is a statement about the stored bits and not about
                 * a tolerance - which is what these cases need, since "close to 0.1" is
                 * true of both the right answer and the wrong one.
                 */
                std::vector<double> read_exact(const Tensor& t)
                {
                    const size_t n = static_cast<size_t>(t.num_elements);
                    std::vector<double> out(n);
                    auto& q = TestEnvironment::queue();

                    if (t.dtype == Core::DataType::Half)
                    {
                        std::vector<sycl::half> host(n);
                        q.memcpy(host.data(), t.data(), n * sizeof(sycl::half)).wait();
                        for (size_t i = 0; i < n; ++i) out[i] = static_cast<double>(host[i]);
                    }
                    else if (t.dtype == Core::DataType::Float32)
                    {
                        std::vector<float> host(n);
                        q.memcpy(host.data(), t.data(), n * sizeof(float)).wait();
                        for (size_t i = 0; i < n; ++i) out[i] = static_cast<double>(host[i]);
                    }
                    else
                    {
                        q.memcpy(out.data(), t.data(), n * sizeof(double)).wait();
                    }
                    return out;
                }

                /** @brief Writes host doubles into a tensor of any of the three real dtypes. */
                void write_exact(Tensor& t, const std::vector<double>& values)
                {
                    const size_t n = values.size();
                    ASSERT_EQ(n, static_cast<size_t>(t.num_elements));
                    auto& q = TestEnvironment::queue();

                    if (t.dtype == Core::DataType::Half)
                    {
                        std::vector<sycl::half> host(n);
                        for (size_t i = 0; i < n; ++i) host[i] = static_cast<sycl::half>(static_cast<float>(values[i]));
                        q.memcpy(t.data(), host.data(), n * sizeof(sycl::half)).wait();
                    }
                    else if (t.dtype == Core::DataType::Float32)
                    {
                        std::vector<float> host(n);
                        for (size_t i = 0; i < n; ++i) host[i] = static_cast<float>(values[i]);
                        q.memcpy(t.data(), host.data(), n * sizeof(float)).wait();
                    }
                    else
                    {
                        q.memcpy(t.data(), values.data(), n * sizeof(double)).wait();
                    }
                }

                /** @brief The message of the `std::exception` @p fn throws, or "" if it does not. */
                template<typename Fn>
                std::string message_of(Fn&& fn)
                {
                    try
                    {
                        fn();
                    }
                    catch (const std::exception& e)
                    {
                        return std::string(e.what());
                    }
                    return std::string();
                }
        };

        /**
         * @brief Tests that fp32 to fp16 lands on the correctly rounded neighbour, not the input.
         * @see tests/unit/README.md
         */
        TEST_F(Unit_ConvertTest, Fp32ToFp16RoundsToTheNearestRepresentableValue)
        {
            auto src = engine->create_tensor({1}, Core::DataType::Float32);
            auto dst = engine->create_tensor({1}, Core::DataType::Half);
            write_exact(src, {static_cast<double>(0.1f)});

            SushiBLAS::ElementwiseOps(*engine).convert(src, dst);
            engine->execute().value();

            const auto got = read_exact(dst);
            EXPECT_EQ(K_TENTH_AS_HALF, got[0])
                << "0.1f must arrive as the binary16 value nearest it, 0.0999755859375.";
            EXPECT_NE(static_cast<double>(0.1f), got[0])
                << "0.1 is not representable in binary16, so a result equal to the fp32 "
                   "input means no conversion happened at all.";
        }

        /**
         * @brief Tests that fp32 to fp16 breaks ties to even.
         * @see tests/unit/README.md
         */
        TEST_F(Unit_ConvertTest, Fp32ToFp16BreaksTiesToEven)
        {
            auto src = engine->create_tensor({2}, Core::DataType::Float32);
            auto dst = engine->create_tensor({2}, Core::DataType::Half);
            write_exact(src, {1.0 + 0.00048828125,        // 1 + 2^-11, exactly halfway
                              1.0 + 0.000732421875});     // 1 + 3*2^-12, past halfway

            SushiBLAS::ElementwiseOps(*engine).convert(src, dst);
            engine->execute().value();

            const auto got = read_exact(dst);
            EXPECT_EQ(1.0, got[0]) << "A tie must resolve to the neighbour with the even mantissa.";
            EXPECT_EQ(1.0 + 0.0009765625, got[1]) << "Past halfway must round up to 1 + 2^-10.";
        }

        /**
         * @brief Tests that an fp32 value past the fp16 range becomes an infinity.
         * @see tests/unit/README.md
         */
        TEST_F(Unit_ConvertTest, Fp32PastTheHalfRangeBecomesInfinity)
        {
            auto src = engine->create_tensor({4}, Core::DataType::Float32);
            auto dst = engine->create_tensor({4}, Core::DataType::Half);
            write_exact(src, {K_HALF_MAX, 70000.0, -70000.0, 1.0e30});

            SushiBLAS::ElementwiseOps(*engine).convert(src, dst);
            engine->execute().value();

            const auto got = read_exact(dst);
            EXPECT_EQ(K_HALF_MAX, got[0]) << "The largest finite binary16 value must stay finite.";
            EXPECT_TRUE(std::isinf(got[1]) && got[1] > 0.0) << "70000 is past binary16's range.";
            EXPECT_TRUE(std::isinf(got[2]) && got[2] < 0.0) << "The sign must survive the overflow.";
            EXPECT_TRUE(std::isinf(got[3]) && got[3] > 0.0);
        }

        /**
         * @brief Tests that 1e-8 flushes to zero in fp16 while 1e-7 lands on a subnormal.
         * @see tests/unit/README.md
         */
        TEST_F(Unit_ConvertTest, Fp32BelowTheHalfRangeUnderflowsToZeroButSubnormalsSurvive)
        {
            auto src = engine->create_tensor({2}, Core::DataType::Float32);
            auto dst = engine->create_tensor({2}, Core::DataType::Half);
            write_exact(src, {1.0e-8, 1.0e-7});

            SushiBLAS::ElementwiseOps(*engine).convert(src, dst);
            engine->execute().value();

            const auto got = read_exact(dst);
            EXPECT_EQ(0.0, got[0]) << "1e-8 is below half the smallest binary16 subnormal.";
            EXPECT_EQ(K_TINY_AS_HALF, got[1])
                << "1e-7 rounds to 2 * 2^-24; flushing it to zero would mean subnormals are "
                   "being discarded rather than represented.";
        }

        /**
         * @brief Tests that infinities and NaN pass through the conversion in both directions.
         * @see tests/unit/README.md
         */
        TEST_F(Unit_ConvertTest, NonFiniteValuesSurviveTheConversion)
        {
            auto src = engine->create_tensor({3}, Core::DataType::Float32);
            auto down = engine->create_tensor({3}, Core::DataType::Half);
            auto up = engine->create_tensor({3}, Core::DataType::Float64);
            write_exact(src, {std::numeric_limits<double>::infinity(),
                              -std::numeric_limits<double>::infinity(),
                              std::numeric_limits<double>::quiet_NaN()});

            SushiBLAS::ElementwiseOps(*engine).convert(src, down);
            SushiBLAS::ElementwiseOps(*engine).convert(src, up);
            engine->execute().value();

            for (const Tensor* t : {&down, &up})
            {
                const auto got = read_exact(*t);
                EXPECT_TRUE(std::isinf(got[0]) && got[0] > 0.0);
                EXPECT_TRUE(std::isinf(got[1]) && got[1] < 0.0);
                EXPECT_TRUE(std::isnan(got[2]));
            }
        }

        /**
         * @brief Tests that narrowing then widening returns the fp16 value, not the original.
         * @see tests/unit/README.md
         */
        TEST_F(Unit_ConvertTest, ARoundTripThroughHalfKeepsOnlyWhatHalfCouldHold)
        {
            auto src = engine->create_tensor({1}, Core::DataType::Float32);
            auto narrow = engine->create_tensor({1}, Core::DataType::Half);
            auto back = engine->create_tensor({1}, Core::DataType::Float32);
            write_exact(src, {static_cast<double>(0.1f)});

            SushiBLAS::ElementwiseOps(*engine).convert(src, narrow);
            SushiBLAS::ElementwiseOps(*engine).convert(narrow, back);
            engine->execute().value();

            const auto got = read_exact(back);
            EXPECT_EQ(K_TENTH_AS_HALF, got[0])
                << "Widening back must reproduce the fp16 value exactly, and must not "
                   "recover precision the narrowing threw away.";
        }

        /**
         * @brief Tests all nine dtype pairs on values every dtype represents exactly.
         * @see tests/unit/README.md
         */
        TEST_F(Unit_ConvertTest, EveryRealDTypePairConvertsExactlyRepresentableValues)
        {
            const std::vector<double> values = {0.0, 1.0, -2.0, 3.5, -4.25, 1024.0};

            for (Core::DataType from : K_REAL_DTYPES)
            {
                for (Core::DataType to : K_REAL_DTYPES)
                {
                    auto src = engine->create_tensor({6}, from);
                    auto dst = engine->create_tensor({6}, to);
                    write_exact(src, values);

                    SushiBLAS::ElementwiseOps(*engine).convert(src, dst);
                    engine->execute().value();

                    const auto got = read_exact(dst);
                    for (size_t i = 0; i < values.size(); ++i)
                    {
                        EXPECT_EQ(values[i], got[i])
                            << "at index " << i << " converting " << Core::to_string(from)
                            << " -> " << Core::to_string(to);
                    }
                }
            }
        }

        /**
         * @brief Tests that converting a dtype to itself copies.
         * @see tests/unit/README.md
         */
        TEST_F(Unit_ConvertTest, ConvertingADTypeToItselfIsAPlainCopy)
        {
            auto src = engine->create_tensor({3}, Core::DataType::Float32);
            auto dst = engine->create_tensor({3}, Core::DataType::Float32);
            // 2^40 is exact in fp32; 1e30 would not be. See tests/unit/README.md.
            write_exact(src, {1.5, -2.25, 1099511627776.0});

            EXPECT_NO_THROW(SushiBLAS::ElementwiseOps(*engine).convert(src, dst));
            engine->execute().value();

            const auto got = read_exact(dst);
            EXPECT_EQ(1.5, got[0]);
            EXPECT_EQ(-2.25, got[1]);
            EXPECT_EQ(1099511627776.0, got[2]);
        }

        // A multi-dimensional operand is converted whole, not just its first row.
        TEST_F(Unit_ConvertTest, ARankTwoOperandIsConvertedInFull)
        {
            auto src = engine->create_tensor({3, 4}, Core::DataType::Float64);
            auto dst = engine->create_tensor({3, 4}, Core::DataType::Half);
            std::vector<double> values(12);
            for (size_t i = 0; i < values.size(); ++i) values[i] = static_cast<double>(i) - 5.0;
            write_exact(src, values);

            SushiBLAS::ElementwiseOps(*engine).convert(src, dst);
            engine->execute().value();

            const auto got = read_exact(dst);
            for (size_t i = 0; i < values.size(); ++i) EXPECT_EQ(values[i], got[i]) << "at index " << i;
        }

        TEST_F(Unit_ConvertTest, ADTypePairOutsideTheRealDomainThrowsNamingBoth)
        {
            GraphProbe probe;

            auto real32 = engine->create_tensor({2}, Core::DataType::Float32);
            auto cplx = engine->create_tensor({2}, Core::DataType::Complex32);
            auto integral = engine->create_tensor({2}, Core::DataType::Int64);

            const std::string from_complex = message_of(
                [&]
                {
                    SushiBLAS::ElementwiseOps(probe.recorder()).convert(cplx, real32);
                });
            EXPECT_NE(std::string::npos, from_complex.find("COMPLEX32")) << from_complex;
            EXPECT_NE(std::string::npos, from_complex.find("FLOAT32")) << from_complex;

            const std::string to_integer = message_of(
                [&]
                {
                    SushiBLAS::ElementwiseOps(probe.recorder()).convert(real32, integral);
                });
            EXPECT_NE(std::string::npos, to_integer.find("FLOAT32")) << to_integer;
            EXPECT_NE(std::string::npos, to_integer.find("INT64")) << to_integer;

            // And nothing was recorded: a rejected conversion must not leave a task behind.
            EXPECT_EQ(0u, probe.take_recorded_count());
        }

        /**
         * @brief Tests that converting between row- and column-major rank-2 tensors throws.
         */
        TEST_F(Unit_ConvertTest, DisagreeingLayoutsThrow)
        {
            GraphProbe probe;

            constexpr int64_t kRows = 2;
            constexpr int64_t kCols = 3;
            constexpr size_t kElements = static_cast<size_t>(kRows * kCols);

            auto f32 = SushiRuntime::make_sushi<Storage>(TestEnvironment::runtime(),
                                                         kElements * sizeof(float),
                                                         SushiRuntime::Residency::Shared);
            auto f16 = SushiRuntime::make_sushi<Storage>(TestEnvironment::runtime(),
                                                         kElements * sizeof(sycl::half),
                                                         SushiRuntime::Residency::Shared);

            Tensor row_major(f32, {kRows, kCols}, Core::DataType::Float32, 0, Core::Layout::RowMajor);
            Tensor col_major(f16, {kRows, kCols}, Core::DataType::Half, 0, Core::Layout::ColumnMajor);

            EXPECT_THROW(
                SushiBLAS::ElementwiseOps(probe.recorder()).convert(row_major, col_major),
                std::invalid_argument);
            EXPECT_EQ(0u, probe.take_recorded_count());
        }

        TEST_F(Unit_ConvertTest, DisagreeingShapesThrowNamingWhatDisagreed)
        {
            auto src = engine->create_tensor({3, 4}, Core::DataType::Float32);
            auto wrong_dim = engine->create_tensor({3, 5}, Core::DataType::Half);
            auto wrong_rank = engine->create_tensor({12}, Core::DataType::Half);

            const std::string dim =
                message_of([&]{ SushiBLAS::ElementwiseOps(*engine).convert(src, wrong_dim); });
            EXPECT_NE(std::string::npos, dim.find("dimension")) << dim;

            const std::string rank =
                message_of([&]{ SushiBLAS::ElementwiseOps(*engine).convert(src, wrong_rank); });
            EXPECT_NE(std::string::npos, rank.find("rank")) << rank;
        }

        /**
         * @brief Tests that an inner-axis slice, which is not contiguous, is refused by name.
         * @see tests/unit/README.md
         */
        TEST_F(Unit_ConvertTest, ANonContiguousOperandIsRefused)
        {
            auto parent = engine->create_tensor({4, 4}, Core::DataType::Float32);
            auto inner = parent.slice(1, 1, 3);
            auto dst = engine->create_tensor({4, 2}, Core::DataType::Half);

            ASSERT_FALSE(inner.is_contiguous());
            const std::string msg =
                message_of([&]{ SushiBLAS::ElementwiseOps(*engine).convert(inner, dst); });
            EXPECT_NE(std::string::npos, msg.find("strided")) << msg;
        }

        /**
         * @brief Tests that overlapping operands are refused.
         * @see tests/unit/README.md
         */
        TEST_F(Unit_ConvertTest, OverlappingOperandsAreRefused)
        {
            constexpr int64_t kN = 8;
            auto storage = SushiRuntime::make_sushi<Storage>(
                TestEnvironment::runtime(),
                static_cast<size_t>(2 * kN) * sizeof(float),
                SushiRuntime::Residency::Shared);

            // Two FLOAT32 views four elements apart over eight elements share the middle four.
            Tensor a(storage, {kN}, Core::DataType::Float32, 0);
            Tensor b(storage, {kN}, Core::DataType::Float32, kN / 2);

            const std::string msg =
                message_of([&]{ SushiBLAS::ElementwiseOps(*engine).convert(a, b); });
            EXPECT_NE(std::string::npos, msg.find("overlap")) << msg;
        }

        /**
         * @brief Tests that a convert is ordered behind the op that wrote its offset source view.
         * @see tests/unit/README.md
         */
        TEST_F(Unit_ConvertTest, AConvertIsOrderedBehindTheOpThatWroteItsSource)
        {
            GraphProbe probe;

            constexpr int64_t kN = 4;
            auto arena = SushiRuntime::make_sushi<Storage>(
                TestEnvironment::runtime(),
                static_cast<size_t>(2 * kN) * sizeof(float),
                SushiRuntime::Residency::Shared);

            Tensor seed(arena, {kN}, Core::DataType::Float32, 0);
            Tensor src(arena, {kN}, Core::DataType::Float32, kN);
            auto dst = engine->create_tensor({kN}, Core::DataType::Half);
            write_exact(seed, {1.0, 2.0, 3.0, 4.0});

            SushiBLAS::ElementwiseOps(probe.recorder()).mul(seed, seed, src);
            SushiBLAS::ElementwiseOps(probe.recorder()).convert(src, dst);

            ASSERT_EQ(2u, probe.recorded_count());
            const auto tasks = probe.recorded_tasks();
            if (!tasks)
                GTEST_SKIP() << GraphProbe::K_NO_RECORDED_TASK_VIEW;
            EXPECT_EQ(1u, (*tasks)[1].dependency_count)
                << "The convert reading `src` must depend on the op that wrote it. An "
                   "in-degree of 0 means the read was declared from a bare pointer rather "
                   "than from Tensor::region(), so the tracker filed it under a resource "
                   "nothing else touches and emitted no edge at all.";
        }

        TEST_F(Unit_ConvertTest, AReaderOfTheConvertedTensorIsOrderedBehindTheConvert)
        {
            GraphProbe probe;

            constexpr int64_t kN = 4;
            auto arena = SushiRuntime::make_sushi<Storage>(
                TestEnvironment::runtime(),
                static_cast<size_t>(2 * kN) * sizeof(float),
                SushiRuntime::Residency::Shared);

            auto src = engine->create_tensor({kN}, Core::DataType::Float32);
            Tensor dst(arena, {kN}, Core::DataType::Float32, kN);
            auto total = engine->create_tensor({1}, Core::DataType::Float32);
            write_exact(src, {1.0, 2.0, 3.0, 4.0});

            SushiBLAS::ElementwiseOps(probe.recorder()).convert(src, dst);
            SushiBLAS::ReductionOps(probe.recorder()).sum(dst, total);

            ASSERT_EQ(2u, probe.recorded_count());
            const auto tasks = probe.recorded_tasks();
            if (!tasks)
                GTEST_SKIP() << GraphProbe::K_NO_RECORDED_TASK_VIEW;
            EXPECT_EQ(1u, (*tasks)[1].dependency_count)
                << "The write set must name `dst` by its own byte interval, so the "
                   "reduction reading it is ordered behind the conversion. `dst` is a view "
                   "partway into an arena, which is where a bare-pointer write set stops "
                   "agreeing with the reader's region.";
        }

        /**
         * @brief Tests that converts over disjoint views of one arena get no edge between them.
         * @see tests/unit/README.md
         */
        TEST_F(Unit_ConvertTest, ConvertsOverDisjointViewsOfOneArenaDoNotSerialise)
        {
            GraphProbe probe;

            constexpr int64_t kN = 8;

            auto f32_arena = SushiRuntime::make_sushi<Storage>(
                TestEnvironment::runtime(),
                static_cast<size_t>(2 * kN) * sizeof(float),
                SushiRuntime::Residency::Shared);
            auto f16_arena = SushiRuntime::make_sushi<Storage>(
                TestEnvironment::runtime(),
                static_cast<size_t>(2 * kN) * sizeof(sycl::half),
                SushiRuntime::Residency::Shared);

            Tensor src_a(f32_arena, {kN}, Core::DataType::Float32, 0);
            Tensor src_b(f32_arena, {kN}, Core::DataType::Float32, kN);
            Tensor dst_a(f16_arena, {kN}, Core::DataType::Half, 0);
            Tensor dst_b(f16_arena, {kN}, Core::DataType::Half, kN);

            SushiBLAS::ElementwiseOps(probe.recorder()).convert(src_a, dst_a);
            SushiBLAS::ElementwiseOps(probe.recorder()).convert(src_b, dst_b);

            ASSERT_EQ(2u, probe.recorded_count());
            const auto tasks = probe.recorded_tasks();
            if (!tasks)
                GTEST_SKIP() << GraphProbe::K_NO_RECORDED_TASK_VIEW;
            EXPECT_EQ(0u, (*tasks)[0].dependency_count);
            EXPECT_EQ(0u, (*tasks)[1].dependency_count)
                << "Two conversions over non-overlapping halves of one arena share no "
                   "bytes, so neither may depend on the other. An in-degree of 1 means the "
                   "operands were described by their allocation rather than by their own "
                   "byte interval, which serialises everything a memory planner packs.";
        }

        /**
         * @brief Tests that a view at a non-zero offset converts its own elements and no others.
         * @see tests/unit/README.md
         */
        TEST_F(Unit_ConvertTest, AViewAtANonZeroOffsetConvertsItsOwnElements)
        {
            constexpr int64_t kN = 4;

            auto f32_arena = SushiRuntime::make_sushi<Storage>(
                TestEnvironment::runtime(),
                static_cast<size_t>(2 * kN) * sizeof(float),
                SushiRuntime::Residency::Shared);
            auto f16_arena = SushiRuntime::make_sushi<Storage>(
                TestEnvironment::runtime(),
                static_cast<size_t>(2 * kN) * sizeof(sycl::half),
                SushiRuntime::Residency::Shared);

            Tensor head(f32_arena, {kN}, Core::DataType::Float32, 0);
            Tensor tail(f32_arena, {kN}, Core::DataType::Float32, kN);
            Tensor out_head(f16_arena, {kN}, Core::DataType::Half, 0);
            Tensor out_tail(f16_arena, {kN}, Core::DataType::Half, kN);

            write_exact(head, {1.0, 2.0, 3.0, 4.0});
            write_exact(tail, {5.0, 6.0, 7.0, 8.0});
            write_exact(out_head, {-1.0, -1.0, -1.0, -1.0});

            SushiBLAS::ElementwiseOps(*engine).convert(tail, out_tail);
            engine->execute().value();

            const auto got_tail = read_exact(out_tail);
            EXPECT_EQ(5.0, got_tail[0]);
            EXPECT_EQ(8.0, got_tail[3]);

            const auto got_head = read_exact(out_head);
            for (size_t i = 0; i < 4; ++i)
            {
                EXPECT_EQ(-1.0, got_head[i])
                    << "Converting the tail view wrote over the head at index " << i
                    << ", which is what dropping storage_offset looks like.";
            }
        }

    } // namespace Test
} // namespace SushiBLAS
