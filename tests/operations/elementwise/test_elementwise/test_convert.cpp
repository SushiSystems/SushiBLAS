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
 * @file test_convert.cpp
 * @brief Tests ElementwiseOps::convert between the real dtypes, its refusals and the regions it declares.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiRuntime/core/sushi_ptr.hpp>
#include <SushiBLAS/tensor/host_block.hpp>
#include <SushiBLAS/engine/host_task_sink.hpp>
#include <SushiBLAS/engine_fixture/host_engine_test.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief The nearest binary16 value to 0.1f, as an exact double: 1.599609375 x 2^-4. */
            constexpr double K_TENTH_AS_HALF = 0.0999755859375;

            /** @brief The largest finite binary16 value. */
            constexpr double K_HALF_MAX = 65504.0;

            /** @brief 2 x 2^-24: the binary16 subnormal nearest to 1e-7. */
            constexpr double K_TINY_AS_HALF = 1.1920928955078125e-7;

            /** @brief 2^-24: one step of the binary16 subnormals, the slack KI-3 allows below 2^-13. */
            constexpr double K_HALF_SUBNORMAL_STEP = 5.9604644775390625e-8;

            /** @brief A positive infinity of double. */
            constexpr double K_INF = std::numeric_limits<double>::infinity();

            /** @brief A quiet NaN of double. */
            constexpr double K_NAN = std::numeric_limits<double>::quiet_NaN();

            /** @brief Returns whether two doubles hold the same value, a NaN equalling a NaN. */
            bool same_value(double expected, double got)
            {
                return (std::isnan(expected) && std::isnan(got)) || expected == got;
            }

            /** @brief Returns the values 0, 1, ..., count - 1 shifted down by five. */
            std::vector<double> ramp(std::size_t count)
            {
                std::vector<double> values(count);
                for (std::size_t i = 0; i < count; ++i)
                    values[i] = static_cast<double>(i) - 5.0;
                return values;
            }
        } // namespace

        /// @brief Records convert on a host engine and runs it on the reference path.
        class Small_ConvertTest : public HostEngineTest
        {
            protected:
                /** @brief Reads a whole real tensor back, widened to double; the widening is exact. */
                std::vector<double> read_exact(const Tensor& t)
                {
                    const std::size_t n = static_cast<std::size_t>(t.num_elements);
                    std::vector<double> out(n);
                    if (t.dtype == Core::DataType::Half)
                    {
                        std::vector<sycl::half> host(n);
                        std::memcpy(host.data(), t.data(), n * sizeof(sycl::half));
                        for (std::size_t i = 0; i < n; ++i)
                            out[i] = static_cast<double>(host[i]);
                    }
                    else if (t.dtype == Core::DataType::Float32)
                    {
                        std::vector<float> host(n);
                        std::memcpy(host.data(), t.data(), n * sizeof(float));
                        for (std::size_t i = 0; i < n; ++i)
                            out[i] = static_cast<double>(host[i]);
                    }
                    else
                    {
                        std::memcpy(out.data(), t.data(), n * sizeof(double));
                    }
                    return out;
                }

                /** @brief Writes host doubles into a tensor of any of the three real dtypes. */
                void write_exact(Tensor& t, const std::vector<double>& values)
                {
                    const std::size_t n = values.size();
                    if (t.dtype == Core::DataType::Half)
                    {
                        std::vector<sycl::half> host(n);
                        for (std::size_t i = 0; i < n; ++i)
                            host[i] = static_cast<sycl::half>(static_cast<float>(values[i]));
                        std::memcpy(t.data(), host.data(), n * sizeof(sycl::half));
                    }
                    else if (t.dtype == Core::DataType::Float32)
                    {
                        std::vector<float> host(n);
                        for (std::size_t i = 0; i < n; ++i)
                            host[i] = static_cast<float>(values[i]);
                        std::memcpy(t.data(), host.data(), n * sizeof(float));
                    }
                    else
                    {
                        std::memcpy(t.data(), values.data(), n * sizeof(double));
                    }
                }

                /** @brief Converts values from one dtype to another and returns what arrived. */
                std::vector<double> converted(Core::DataType from, Core::DataType to,
                                              const std::vector<double>& values)
                {
                    const int64_t n = static_cast<int64_t>(values.size());
                    Tensor src = engine->create_tensor({n}, from);
                    Tensor dst = engine->create_tensor({n}, to);
                    write_exact(src, values);

                    ElementwiseOps(recorder()).convert(src, dst);
                    EXPECT_TRUE(recording->run_succeeds());
                    return read_exact(dst);
                }

                /** @brief Returns the message of the std::invalid_argument fn throws, or "". */
                template<typename Fn>
                std::string refusal_of(Fn&& fn)
                {
                    try
                    {
                        fn();
                    }
                    catch (const std::invalid_argument& error)
                    {
                        return error.what();
                    }
                    return std::string();
                }

                /** @brief Returns a FLOAT32 vector view at an element offset of a host arena. */
                Tensor view_of(const SushiRuntime::sushi_ptr<Storage>& arena, Core::DataType dtype, int64_t n,
                               int64_t offset)
                {
                    return Tensor(arena, {n}, dtype, offset);
                }
        };

        /** @brief One FLOAT32 value converted to HALF and the binary16 value it must arrive as. */
        struct ValueRow
        {
            std::string name;
            double input;
            double expected;
            /// @brief How far the host's conversion may land from expected; 0 asks for the exact value.
            double slack = 0.0;
        };

        /** @brief Names a row by its name. */
        std::string value_row_name(const ::testing::TestParamInfo<ValueRow>& info)
        {
            return info.param.name;
        }

        /// @brief Runs the rows whose FLOAT32 input lands on a representable HALF neighbour.
        class Small_ConvertRounding : public Small_ConvertTest, public ::testing::WithParamInterface<ValueRow> {};

        /// @brief Runs the rows whose FLOAT32 input is past the HALF range.
        class Small_ConvertOverflow : public Small_ConvertTest, public ::testing::WithParamInterface<ValueRow> {};

        /// @brief Runs the rows whose FLOAT32 input is below the HALF range.
        class Small_ConvertUnderflow : public Small_ConvertTest, public ::testing::WithParamInterface<ValueRow> {};

        /// @brief Tests that fp32 to fp16 lands on the neighbour binary16 rounding gives, ties to even.
        TEST_P(Small_ConvertRounding, Fp32ToFp16LandsOnTheRoundedNeighbour)
        {
            const std::vector<double> got =
                converted(Core::DataType::Float32, Core::DataType::Half, {GetParam().input});

            EXPECT_EQ(GetParam().expected, got[0]);
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_ConvertRounding,
            ::testing::Values(
                // 0.1f has no binary16 form; the first dropped mantissa bit is 0, so it rounds down.
                ValueRow{"ATenthRoundsDown", static_cast<double>(0.1f), K_TENTH_AS_HALF},
                // 1 + 2^-11 is exactly halfway between 1 and 1 + 2^-10; the even mantissa is 1.
                ValueRow{"ATieGoesToTheEvenNeighbour", 1.0 + 0.00048828125, 1.0},
                // 1 + 3 x 2^-12 is past halfway, so it rounds up to 1 + 2^-10.
                ValueRow{"PastHalfwayRoundsUp", 1.0 + 0.000732421875, 1.0 + 0.0009765625}),
            value_row_name);

        /// @brief Tests that an fp32 value past the fp16 range becomes an infinity of its sign.
        TEST_P(Small_ConvertOverflow, Fp32PastTheHalfRangeBecomesAnInfinity)
        {
            const std::vector<double> got =
                converted(Core::DataType::Float32, Core::DataType::Half, {GetParam().input});

            EXPECT_EQ(GetParam().expected, got[0]);
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_ConvertOverflow,
            ::testing::Values(ValueRow{"TheLargestHalfStaysFinite", K_HALF_MAX, K_HALF_MAX},
                              ValueRow{"SeventyThousandBecomesInfinity", 70000.0, K_INF},
                              ValueRow{"MinusSeventyThousandBecomesMinusInfinity", -70000.0, -K_INF},
                              ValueRow{"OneEThirtyBecomesInfinity", 1.0e30, K_INF}),
            value_row_name);

        /// @brief Tests that an fp32 value below the fp16 range flushes to zero and a subnormal survives.
        TEST_P(Small_ConvertUnderflow, Fp32BelowTheHalfRangeUnderflowsOrLandsOnASubnormal)
        {
            const std::vector<double> got =
                converted(Core::DataType::Float32, Core::DataType::Half, {GetParam().input});

            // KI-3: the host truncates below 2^-13, so a row states one subnormal step as its slack.
            EXPECT_LE(std::abs(GetParam().expected - got[0]), GetParam().slack);
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_ConvertUnderflow,
            ::testing::Values(
                // 1e-8 is below half the smallest subnormal, 2^-25.
                ValueRow{"OneEMinusEightFlushesToZero", 1.0e-8, 0.0},
                // 1e-7 / 2^-24 is 1.677, which rounds to 2.
                ValueRow{"OneEMinusSevenLandsOnASubnormal", 1.0e-7, K_TINY_AS_HALF,
                         K_HALF_SUBNORMAL_STEP}),
            value_row_name);

        /** @brief A non-finite FLOAT32 value converted to a target dtype. */
        struct NonFiniteRow
        {
            std::string name;
            Core::DataType target;
            double value;
        };

        /** @brief Names a row by its name. */
        std::string non_finite_row_name(const ::testing::TestParamInfo<NonFiniteRow>& info)
        {
            return info.param.name;
        }

        /// @brief Runs the rows of non-finite values over the two targets.
        class Small_ConvertNonFinite : public Small_ConvertTest, public ::testing::WithParamInterface<NonFiniteRow> {};

        /// @brief Tests that an infinity or a NaN keeps its kind through the conversion.
        TEST_P(Small_ConvertNonFinite, ANonFiniteValueSurvivesTheConversion)
        {
            const std::vector<double> got =
                converted(Core::DataType::Float32, GetParam().target, {GetParam().value});

            EXPECT_TRUE(same_value(GetParam().value, got[0]));
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_ConvertNonFinite,
            ::testing::Values(NonFiniteRow{"InfinityToHalf", Core::DataType::Half, K_INF},
                              NonFiniteRow{"MinusInfinityToHalf", Core::DataType::Half, -K_INF},
                              NonFiniteRow{"NaNToHalf", Core::DataType::Half, K_NAN},
                              NonFiniteRow{"InfinityToFloat64", Core::DataType::Float64, K_INF},
                              NonFiniteRow{"MinusInfinityToFloat64", Core::DataType::Float64, -K_INF},
                              NonFiniteRow{"NaNToFloat64", Core::DataType::Float64, K_NAN}),
            non_finite_row_name);

        /// @brief Tests that narrowing then widening returns the fp16 value, not the original.
        TEST_F(Small_ConvertTest, ARoundTripThroughHalfKeepsOnlyWhatHalfCouldHold)
        {
            const std::vector<double> narrow =
                converted(Core::DataType::Float32, Core::DataType::Half, {static_cast<double>(0.1f)});
            const std::vector<double> back = converted(Core::DataType::Half, Core::DataType::Float32, narrow);

            EXPECT_EQ(K_TENTH_AS_HALF, back[0]);
        }

        /** @brief A source and a target dtype of the nine real pairs. */
        struct PairRow
        {
            std::string name;
            Core::DataType from;
            Core::DataType to;
        };

        /** @brief Names a row by its name. */
        std::string pair_row_name(const ::testing::TestParamInfo<PairRow>& info)
        {
            return info.param.name;
        }

        /// @brief Runs one row per real dtype pair.
        class Small_ConvertPair : public Small_ConvertTest, public ::testing::WithParamInterface<PairRow> {};

        /// @brief Tests that values every real dtype represents exactly arrive unchanged in each pair.
        TEST_P(Small_ConvertPair, ExactlyRepresentableValuesArriveUnchanged)
        {
            const std::vector<double> values = {0.0, 1.0, -2.0, 3.5, -4.25, 1024.0};

            EXPECT_EQ(values, converted(GetParam().from, GetParam().to, values));
        }

        INSTANTIATE_TEST_SUITE_P(
            , Small_ConvertPair,
            ::testing::Values(PairRow{"HalfToHalf", Core::DataType::Half, Core::DataType::Half},
                              PairRow{"HalfToFloat32", Core::DataType::Half, Core::DataType::Float32},
                              PairRow{"HalfToFloat64", Core::DataType::Half, Core::DataType::Float64},
                              PairRow{"Float32ToHalf", Core::DataType::Float32, Core::DataType::Half},
                              PairRow{"Float32ToFloat32", Core::DataType::Float32, Core::DataType::Float32},
                              PairRow{"Float32ToFloat64", Core::DataType::Float32, Core::DataType::Float64},
                              PairRow{"Float64ToHalf", Core::DataType::Float64, Core::DataType::Half},
                              PairRow{"Float64ToFloat32", Core::DataType::Float64, Core::DataType::Float32},
                              PairRow{"Float64ToFloat64", Core::DataType::Float64, Core::DataType::Float64}),
            pair_row_name);

        /// @brief Tests that converting a dtype to itself copies a value fp32 holds only at that width.
        TEST_F(Small_ConvertTest, ConvertingADTypeToItselfIsAPlainCopy)
        {
            Tensor src = engine->create_tensor({3}, Core::DataType::Float32);
            Tensor dst = engine->create_tensor({3}, Core::DataType::Float32);
            // 2^40 is exact in fp32; 1e30 would not be.
            write_exact(src, {1.5, -2.25, 1099511627776.0});

            EXPECT_NO_THROW(ElementwiseOps(recorder()).convert(src, dst));
            ASSERT_TRUE(recording->run_succeeds());

            EXPECT_EQ((std::vector<double>{1.5, -2.25, 1099511627776.0}), read_exact(dst));
        }

        /// @brief Tests that a rank two operand is converted whole, not just its first row.
        TEST_F(Small_ConvertTest, ARankTwoOperandIsConvertedInFull)
        {
            const std::vector<double> values = ramp(12);
            Tensor src = engine->create_tensor({3, 4}, Core::DataType::Float64);
            Tensor dst = engine->create_tensor({3, 4}, Core::DataType::Half);
            write_exact(src, values);

            ElementwiseOps(recorder()).convert(src, dst);
            ASSERT_TRUE(recording->run_succeeds());

            EXPECT_EQ(values, read_exact(dst));
        }

        /// @brief Tests that a complex source is refused, naming both dtypes.
        TEST_F(Small_ConvertTest, AComplexSourceIsRefusedNamingBothDTypes)
        {
            Tensor real32 = engine->create_tensor({2}, Core::DataType::Float32);
            Tensor cplx = engine->create_tensor({2}, Core::DataType::Complex32);

            const std::string message = refusal_of([&] { ElementwiseOps(recorder()).convert(cplx, real32); });

            EXPECT_NE(std::string::npos, message.find("COMPLEX32")) << message;
            EXPECT_EQ(recording->recorded_count(), 0u);
        }

        /// @brief Tests that an integer target is refused, naming its dtype.
        TEST_F(Small_ConvertTest, AnIntegerTargetIsRefusedNamingItsDType)
        {
            Tensor real32 = engine->create_tensor({2}, Core::DataType::Float32);
            Tensor integral = engine->create_tensor({2}, Core::DataType::Int64);

            const std::string message =
                refusal_of([&] { ElementwiseOps(recorder()).convert(real32, integral); });

            EXPECT_NE(std::string::npos, message.find("INT64")) << message;
            EXPECT_EQ(recording->recorded_count(), 0u);
        }

        /// @brief Tests that a row-major source and a column-major target are refused.
        TEST_F(Small_ConvertTest, DisagreeingLayoutsAreRefused)
        {
            constexpr int64_t K_ROWS = 2;
            constexpr int64_t K_COLUMNS = 3;
            constexpr std::size_t K_ELEMENTS = static_cast<std::size_t>(K_ROWS * K_COLUMNS);
            Tensor row_major(make_host_storage(K_ELEMENTS * sizeof(float)), {K_ROWS, K_COLUMNS},
                             Core::DataType::Float32, 0, Core::Layout::RowMajor);
            Tensor col_major(make_host_storage(K_ELEMENTS * sizeof(sycl::half)), {K_ROWS, K_COLUMNS},
                             Core::DataType::Half, 0, Core::Layout::ColumnMajor);

            EXPECT_THROW(ElementwiseOps(recorder()).convert(row_major, col_major), std::invalid_argument);
            EXPECT_EQ(recording->recorded_count(), 0u);
        }

        /// @brief Tests that a target of another extent is refused, naming the dimension.
        TEST_F(Small_ConvertTest, ATargetOfAnotherExtentIsRefusedNamingTheDimension)
        {
            Tensor src = engine->create_tensor({3, 4}, Core::DataType::Float32);
            Tensor wrong_dim = engine->create_tensor({3, 5}, Core::DataType::Half);

            const std::string message = refusal_of([&] { ElementwiseOps(recorder()).convert(src, wrong_dim); });

            EXPECT_NE(std::string::npos, message.find("dimension")) << message;
        }

        /// @brief Tests that a target of another rank is refused, naming the rank.
        TEST_F(Small_ConvertTest, ATargetOfAnotherRankIsRefusedNamingTheRank)
        {
            Tensor src = engine->create_tensor({3, 4}, Core::DataType::Float32);
            Tensor wrong_rank = engine->create_tensor({12}, Core::DataType::Half);

            const std::string message = refusal_of([&] { ElementwiseOps(recorder()).convert(src, wrong_rank); });

            EXPECT_NE(std::string::npos, message.find("rank")) << message;
        }

        /// @brief Tests that an inner-axis slice, which is not contiguous, is refused as strided.
        TEST_F(Small_ConvertTest, ANonContiguousOperandIsRefused)
        {
            Tensor parent = engine->create_tensor({4, 4}, Core::DataType::Float32);
            Tensor inner = parent.slice(1, 1, 3);
            Tensor dst = engine->create_tensor({4, 2}, Core::DataType::Half);

            ASSERT_FALSE(inner.is_contiguous());
            const std::string message = refusal_of([&] { ElementwiseOps(recorder()).convert(inner, dst); });

            EXPECT_NE(std::string::npos, message.find("strided")) << message;
        }

        /// @brief Tests that two views sharing bytes are refused as overlapping.
        TEST_F(Small_ConvertTest, OverlappingOperandsAreRefused)
        {
            constexpr int64_t K_N = 8;
            auto storage = make_host_storage(static_cast<std::size_t>(2 * K_N) * sizeof(float));
            // Two FLOAT32 views four elements apart over eight elements share the middle four.
            Tensor a = view_of(storage, Core::DataType::Float32, K_N, 0);
            Tensor b = view_of(storage, Core::DataType::Float32, K_N, K_N / 2);

            const std::string message = refusal_of([&] { ElementwiseOps(recorder()).convert(a, b); });

            EXPECT_NE(std::string::npos, message.find("overlap")) << message;
        }

        /// @brief Tests that a convert declares the byte interval of its offset source view as a read.
        TEST_F(Small_ConvertTest, AConvertReadsTheByteIntervalOfItsSource)
        {
            constexpr int64_t K_N = 4;
            auto arena = make_host_storage(static_cast<std::size_t>(2 * K_N) * sizeof(float));
            Tensor seed = view_of(arena, Core::DataType::Float32, K_N, 0);
            Tensor src = view_of(arena, Core::DataType::Float32, K_N, K_N);
            Tensor dst = engine->create_tensor({K_N}, Core::DataType::Half);

            ElementwiseOps(recorder()).convert(src, dst);

            const std::vector<HostTaskRecord> records = recording->records();
            ASSERT_EQ(records.size(), 1u);
            ASSERT_EQ(records[0].reads.size(), 1u);
            EXPECT_TRUE(records[0].reads[0].overlaps(src.region()));
            EXPECT_FALSE(records[0].reads[0].overlaps(seed.region()));
        }

        /// @brief Tests that a convert declares the byte interval of its offset target view as a write.
        TEST_F(Small_ConvertTest, AConvertWritesTheByteIntervalOfItsTarget)
        {
            constexpr int64_t K_N = 4;
            auto arena = make_host_storage(static_cast<std::size_t>(2 * K_N) * sizeof(float));
            Tensor head = view_of(arena, Core::DataType::Float32, K_N, 0);
            Tensor src = engine->create_tensor({K_N}, Core::DataType::Float32);
            Tensor dst = view_of(arena, Core::DataType::Float32, K_N, K_N);

            ElementwiseOps(recorder()).convert(src, dst);

            const std::vector<HostTaskRecord> records = recording->records();
            ASSERT_EQ(records.size(), 1u);
            ASSERT_EQ(records[0].writes.size(), 1u);
            EXPECT_TRUE(records[0].writes[0].overlaps(dst.region()));
            EXPECT_FALSE(records[0].writes[0].overlaps(head.region()));
        }

        /// @brief Tests that converts over disjoint views of one arena declare regions that share no bytes.
        TEST_F(Small_ConvertTest, ConvertsOverDisjointViewsOfOneArenaDeclareNoSharedBytes)
        {
            constexpr int64_t K_N = 8;
            auto f32_arena = make_host_storage(static_cast<std::size_t>(2 * K_N) * sizeof(float));
            auto f16_arena = make_host_storage(static_cast<std::size_t>(2 * K_N) * sizeof(sycl::half));
            Tensor src_a = view_of(f32_arena, Core::DataType::Float32, K_N, 0);
            Tensor src_b = view_of(f32_arena, Core::DataType::Float32, K_N, K_N);
            Tensor dst_a = view_of(f16_arena, Core::DataType::Half, K_N, 0);
            Tensor dst_b = view_of(f16_arena, Core::DataType::Half, K_N, K_N);

            ElementwiseOps(recorder()).convert(src_a, dst_a);
            ElementwiseOps(recorder()).convert(src_b, dst_b);

            const std::vector<HostTaskRecord> records = recording->records();
            ASSERT_EQ(records.size(), 2u);
            EXPECT_FALSE(records[0].reads[0].overlaps(records[1].reads[0]));
            EXPECT_FALSE(records[0].writes[0].overlaps(records[1].writes[0]));
            EXPECT_FALSE(records[0].reads[0].overlaps(records[1].writes[0]));
            EXPECT_FALSE(records[0].writes[0].overlaps(records[1].reads[0]));
        }

        /// @brief Tests that a view at a non-zero offset converts its own elements.
        TEST_F(Small_ConvertTest, AViewAtANonZeroOffsetConvertsItsOwnElements)
        {
            constexpr int64_t K_N = 4;
            auto f32_arena = make_host_storage(static_cast<std::size_t>(2 * K_N) * sizeof(float));
            auto f16_arena = make_host_storage(static_cast<std::size_t>(2 * K_N) * sizeof(sycl::half));
            Tensor tail = view_of(f32_arena, Core::DataType::Float32, K_N, K_N);
            Tensor out_tail = view_of(f16_arena, Core::DataType::Half, K_N, K_N);
            write_exact(tail, {5.0, 6.0, 7.0, 8.0});

            ElementwiseOps(recorder()).convert(tail, out_tail);
            ASSERT_TRUE(recording->run_succeeds());

            EXPECT_EQ((std::vector<double>{5.0, 6.0, 7.0, 8.0}), read_exact(out_tail));
        }

        /// @brief Tests that converting the tail view leaves the head bytes of the target arena as they were.
        TEST_F(Small_ConvertTest, AViewAtANonZeroOffsetLeavesTheHeadUntouched)
        {
            constexpr int64_t K_N = 4;
            auto f32_arena = make_host_storage(static_cast<std::size_t>(2 * K_N) * sizeof(float));
            auto f16_arena = make_host_storage(static_cast<std::size_t>(2 * K_N) * sizeof(sycl::half));
            Tensor tail = view_of(f32_arena, Core::DataType::Float32, K_N, K_N);
            Tensor out_head = view_of(f16_arena, Core::DataType::Half, K_N, 0);
            Tensor out_tail = view_of(f16_arena, Core::DataType::Half, K_N, K_N);
            write_exact(tail, {5.0, 6.0, 7.0, 8.0});
            write_exact(out_head, {-1.0, -1.0, -1.0, -1.0});

            ElementwiseOps(recorder()).convert(tail, out_tail);
            ASSERT_TRUE(recording->run_succeeds());

            // A conversion that dropped storage_offset would write over the head.
            EXPECT_EQ((std::vector<double>{-1.0, -1.0, -1.0, -1.0}), read_exact(out_head));
        }
    } // namespace Test
} // namespace SushiBLAS
