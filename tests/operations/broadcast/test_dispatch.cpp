/**************************************************************************/
/* test_dispatch.cpp                                                      */
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
 * @file test_dispatch.cpp
 * @brief Tests Dispatch::execute_broadcast with several vectors and a bound scalar, its refusals,
 *        and that the host path equals the device bit for bit.
 * @author Mustafa Garip
 */

#include <array>
#include <algorithm>
#include <string>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/broadcast/dispatch.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/scalar_dtype.hpp>
#include <SushiBLAS/tensor_expectations/tensor_readback.hpp>
#include <SushiBLAS/typed_suite_naming/default_type_names.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
#include <SushiRuntime/core/operation_id.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        using namespace SushiRuntime::Literals;

        namespace
        {
            /// @brief The name every call of this file records under.
            constexpr const char* K_NAME = "test.broadcast_seam";

            /// @brief The scalar the mixed call binds.
            constexpr double K_SCALE = 2.0;

            /**
             * @brief Returns count values that are multiples of 0.25 between -0.5 and 2.0.
             * @param seed Shifts the sequence, so two vectors of one call differ.
             */
            template<typename T>
            std::vector<T> quarters(std::size_t count, std::size_t seed = 3)
            {
                constexpr std::size_t K_PERIOD = 11;
                std::vector<T> values;
                values.reserve(count);
                for (std::size_t i = 0; i < count; ++i)
                    values.push_back(static_cast<T>(
                        0.25 * (static_cast<double>((i * 5 + seed) % K_PERIOD) - 2.0)));
                return values;
            }

            /// @brief Returns count values whose division is not exact: 1 / (i + 3) and 7 / (i + 3).
            template<typename T>
            std::vector<T> inexact(std::size_t count, double numerator)
            {
                std::vector<T> values;
                values.reserve(count);
                for (std::size_t i = 0; i < count; ++i)
                    values.push_back(static_cast<T>(numerator / (static_cast<double>(i) + 3.0)));
                return values;
            }

            /// @brief Returns the bytes of the elements, so two results compare bit for bit.
            template<typename T>
            std::vector<unsigned char> bytes_of(const std::vector<T>& values)
            {
                std::vector<unsigned char> bytes(values.size() * sizeof(T));
                std::memcpy(bytes.data(), values.data(), bytes.size());
                return bytes;
            }

            /// @brief Returns the vector widened to double.
            template<typename T>
            std::vector<double> widened(const std::vector<T>& values)
            {
                return std::vector<double>(values.begin(), values.end());
            }

            /// @brief Returns whether the test device can compute in T.
            template<typename T>
            bool device_computes()
            {
                const sycl::device device = TestEnvironment::queue().get_device();

                if constexpr (std::is_same_v<T, sycl::half>)
                    return device.has(sycl::aspect::fp16);
                else if constexpr (std::is_same_v<T, double>)
                    return device.has(sycl::aspect::fp64);
                else
                    return true;
            }

            /**
             * @brief Expects record to throw std::invalid_argument whose message holds expected.
             * @tparam Record Callable `() -> void` that records one malformed call.
             */
            template<typename Record>
            void expect_refusal(Record&& record, const std::string& expected)
            {
                try
                {
                    record();
                    ADD_FAILURE() << "the call was accepted; it should name '" << expected << "'";
                }
                catch (const std::invalid_argument& error)
                {
                    const std::string text = error.what();
                    EXPECT_NE(text.find(K_NAME), std::string::npos) << text;
                    EXPECT_NE(text.find(expected), std::string::npos) << text;
                }
            }

            /**
             * @brief Computes the mixed call on the host: `(x - l[o]) * t0[n] + t1[n] * scale`.
             * @param outer The extent of the leading vector.
             * @param inner The extent of the trailing vectors.
             */
            std::vector<float> mixed_on_the_host(const std::vector<float>& a,
                                                 const std::vector<float>& leading,
                                                 const std::vector<float>& first,
                                                 const std::vector<float>& second,
                                                 std::size_t outer, std::size_t inner)
            {
                std::vector<float> expected(outer * inner);
                for (std::size_t o = 0; o < outer; ++o)
                {
                    for (std::size_t n = 0; n < inner; ++n)
                    {
                        const std::size_t i = o * inner + n;
                        expected[i] = (a[i] - leading[o]) * first[n] +
                                      second[n] * static_cast<float>(K_SCALE);
                    }
                }
                return expected;
            }

            /**
             * @brief Adds the window of source that starts at offset to each row of a, on the host.
             * @param width The extent of a row, which is the length of the window.
             */
            std::vector<float> plus_window_on_the_host(const std::vector<float>& a,
                                                       const std::vector<float>& source,
                                                       std::size_t offset, std::size_t width)
            {
                std::vector<float> expected(a);
                for (std::size_t i = 0; i < expected.size(); ++i)
                    expected[i] += source[offset + i % width];
                return expected;
            }

            /** @brief Holds one malformed geometry: the matrix, and the counts of its vectors. */
            struct BadCounts
            {
                /** @brief Names the case in the test name. */
                const char* label;
                /** @brief Holds the extents of A. */
                std::vector<int64_t> dims;
                /** @brief Holds the element count of the leading vector, or 0 for none. */
                int64_t leading;
                /** @brief Holds the element count of the trailing vector, or 0 for none. */
                int64_t trailing;
            };
        } // namespace

        /** @brief Gives each test the fixture's engine, and builders for the tensors it records over. */
        class Medium_BroadcastSeam : public SushiBLASTest
        {
            protected:
                /** @brief Creates a tensor of the given extents holding values. */
                template<typename T>
                Tensor filled(const std::vector<int64_t>& dims, const std::vector<T>& values)
                {
                    Tensor t = engine->create_tensor(
                        SushiRuntime::span<const int64_t>(dims.data(), dims.size()),
                        sushiblas_dtype<T>());
                    TestEnvironment::queue()
                        .memcpy(t.data(), values.data(), values.size() * sizeof(T))
                        .wait();
                    return t;
                }

                /** @brief Creates a tensor of the extents and dtype of t, holding nothing yet. */
                Tensor like(const Tensor& t)
                {
                    return engine->create_tensor(
                        SushiRuntime::span<const int64_t>(t.shape.data(),
                                                          static_cast<std::size_t>(t.rank)),
                        t.dtype);
                }

                /** @brief Records the call under the mode and runs it. */
                template<typename Record>
                void run_in(ExecutionMode mode, Record&& record)
                {
                    set_mode(mode);
                    record();
                    recording->run().value();
                }

                /** @brief Records `C = A + v0 + ... + vN-1` through the seam. */
                template<std::size_t N>
                TaskHandle record_sum(const Tensor& A, const std::array<BroadcastOperand, N>& vectors,
                                      Tensor& C)
                {
                    return Dispatch::execute_broadcast(
                        recorder(), A, vectors, C, K_NAME, "test.broadcast_seam"_op,
                        [](auto x, auto... v) { return (x + ... + v); });
                }

                /** @brief Creates a vector of count elements, or of one when count is 0. */
                Tensor vector_of(std::int64_t count)
                {
                    const std::int64_t length = std::max<std::int64_t>(count, 1);
                    return filled<float>({length}, quarters<float>(static_cast<std::size_t>(length)));
                }

                /** @brief Records the sum of A and the leading and trailing vectors the counts name. */
                void record_counts(const BadCounts& counts, const Tensor& a, Tensor& c)
                {
                    const Tensor leading = vector_of(counts.leading);
                    const Tensor trailing = vector_of(counts.trailing);
                    if (counts.leading > 0 && counts.trailing > 0)
                        record_sum(a, std::array<BroadcastOperand, 2>{
                                          {{&leading, BroadcastAxis::Leading},
                                           {&trailing, BroadcastAxis::Trailing}}}, c);
                    else if (counts.leading > 0)
                        record_sum(a, std::array<BroadcastOperand, 1>{
                                          {{&leading, BroadcastAxis::Leading}}}, c);
                    else
                        record_sum(a, std::array<BroadcastOperand, 1>{
                                          {{&trailing, BroadcastAxis::Trailing}}}, c);
                }

                /** @brief Returns a copy of t that claims the column-major layout. */
                static Tensor as_column_major(const Tensor& t)
                {
                    Tensor copy = t;
                    copy.layout = Core::Layout::ColumnMajor;
                    return copy;
                }
        };

        /** @brief Runs a case under each forced mode. */
        class Medium_BroadcastSeamModes
            : public Medium_BroadcastSeam,
              public ::testing::WithParamInterface<ExecutionMode>
        {
        };

        /** @brief Runs a case in one element type, skipped where the device lacks it. */
        template<typename T>
        class Medium_BroadcastSeamTyped : public Medium_BroadcastSeam
        {
            protected:
                /** @brief Builds the engine and skips the case on a device without T's aspect. */
                void SetUp() override
                {
                    Medium_BroadcastSeam::SetUp();

                    if (!device_computes<T>())
                        GTEST_SKIP() << "the device does not report the aspect this dtype needs";
                }
        };

        using BroadcastSeamTypes = ::testing::Types<sycl::half, float, double>;
        SB_TYPED_TEST_SUITE(Medium_BroadcastSeamTyped, BroadcastSeamTypes);

        /** @brief Runs a refusal case over a table of malformed counts. */
        class Medium_BroadcastSeamCounts : public Medium_BroadcastSeam,
                                           public ::testing::WithParamInterface<BadCounts>
        {
        };

        TEST_P(Medium_BroadcastSeamModes,
               OneLeadingAndTwoTrailingVectorsWithBoundScalarsMatchAHostLoop)
        {
            constexpr std::int64_t K_OUTER = 3;
            constexpr std::int64_t K_INNER = 20;
            const std::vector<float> a_values = quarters<float>(K_OUTER * K_INNER);
            const std::vector<float> leading = quarters<float>(K_OUTER);
            const std::vector<float> first = quarters<float>(K_INNER);
            const std::vector<float> second = quarters<float>(K_INNER, 7);

            // A is [3, 4, 5]; the first trailing vector is a [4, 5] table, the second a flat [20].
            const Tensor a = filled<float>({K_OUTER, 4, 5}, a_values);
            const Tensor v_leading = filled<float>({K_OUTER}, leading);
            const Tensor v_first = filled<float>({4, 5}, first);
            const Tensor v_second = filled<float>({K_INNER}, second);
            Tensor c = like(a);

            const std::array<BroadcastOperand, 3> vectors{
                {{&v_leading, BroadcastAxis::Leading},
                 {&v_first, BroadcastAxis::Trailing},
                 {&v_second, BroadcastAxis::Trailing}}};
            run_in(GetParam(), [&]()
            {
                Dispatch::execute_broadcast(
                    recorder(), a, vectors, c, K_NAME, "test.broadcast_seam"_op,
                    Dispatch::bind_scalars(
                        [](auto x, auto l, auto t, auto u, auto s) { return (x - l) * t + u * s; },
                        Dispatch::ElementScalar(K_SCALE)));
            });

            EXPECT_EQ(read_tensor<float>(c),
                      mixed_on_the_host(a_values, leading, first, second, K_OUTER, K_INNER));
        }

        INSTANTIATE_TEST_SUITE_P(
            Modes, Medium_BroadcastSeamModes,
            ::testing::Values(ExecutionMode::ForceReference, ExecutionMode::ForceSYCL),
            [](const ::testing::TestParamInfo<ExecutionMode>& info)
            {
                return std::string(info.param == ExecutionMode::ForceReference ? "Reference"
                                                                               : "Device");
            });

        TYPED_TEST(Medium_BroadcastSeamTyped, TheReferenceEqualsTheDeviceBitForBit)
        {
            using T = TypeParam;
            const std::vector<T> a_values = inexact<T>(24, 7.0);
            const std::vector<T> leading = inexact<T>(4, 5.0);
            const std::vector<T> trailing = inexact<T>(6, 1.0);
            const Tensor a = this->template filled<T>({4, 6}, a_values);
            const Tensor v_leading = this->template filled<T>({4}, leading);
            const Tensor v_trailing = this->template filled<T>({6}, trailing);
            Tensor reference = this->like(a);
            Tensor device = this->like(a);
            const std::array<BroadcastOperand, 2> vectors{
                {{&v_leading, BroadcastAxis::Leading}, {&v_trailing, BroadcastAxis::Trailing}}};
            const auto record = [&](Tensor& c)
            {
                Dispatch::execute_broadcast(this->recorder(), a, vectors, c, K_NAME,
                                            "test.broadcast_seam"_op,
                                            [](auto x, auto l, auto t) { return x / l + t; });
            };

            this->run_in(ExecutionMode::ForceReference, [&]() { record(reference); });
            this->run_in(ExecutionMode::ForceSYCL, [&]() { record(device); });

            EXPECT_EQ(bytes_of(read_tensor<T>(reference)), bytes_of(read_tensor<T>(device)));
        }

        TEST_F(Medium_BroadcastSeam, AVectorThatIsAViewIntoAnArenaIsReadAtItsOffset)
        {
            const std::vector<float> arena_values = quarters<float>(32);
            const std::vector<float> a_values = quarters<float>(24);
            const Tensor arena = filled<float>({32}, arena_values);
            const Tensor v = arena.slice(0, 7, 13);
            const Tensor a = filled<float>({4, 6}, a_values);
            Tensor c = like(a);
            const std::array<BroadcastOperand, 1> vectors{{{&v, BroadcastAxis::Trailing}}};

            run_in(ExecutionMode::ForceSYCL, [&]() { record_sum(a, vectors, c); });

            EXPECT_EQ(read_tensor<float>(c), plus_window_on_the_host(a_values, arena_values, 7, 6));
        }

        TEST_F(Medium_BroadcastSeam, AnEmptyMatrixRecordsNothingAndReturnsAnEmptyHandle)
        {
            const Tensor a = engine->create_tensor({0, 3}, Core::DataType::Float32);
            const Tensor v = filled<float>({3}, quarters<float>(3));
            Tensor c = engine->create_tensor({0, 3}, Core::DataType::Float32);
            const std::array<BroadcastOperand, 1> vectors{{{&v, BroadcastAxis::Trailing}}};

            const TaskHandle handle = record_sum(a, vectors, c);

            EXPECT_FALSE(handle.is_valid());
            EXPECT_EQ(recording->recorded_count(), std::size_t{0});
        }

        TEST_P(Medium_BroadcastSeamCounts, ACountThatFitsNoPrefixOfTheShapeIsRefused)
        {
            const BadCounts& bad = GetParam();
            const Tensor a = filled<float>(
                bad.dims, quarters<float>(static_cast<std::size_t>(bad.dims[0] * bad.dims[1])));
            Tensor c = like(a);

            expect_refusal([&]() { record_counts(bad, a, c); }, "vectors");
        }

        INSTANTIATE_TEST_SUITE_P(
            Counts, Medium_BroadcastSeamCounts,
            ::testing::Values(BadCounts{"LeadingCountIsNoPrefixProduct", {3, 4}, 2, 0},
                              BadCounts{"LeadingCountDoesNotDivide", {3, 4}, 5, 0},
                              BadCounts{"TrailingCountLeavesNoPrefixProduct", {3, 4}, 0, 6},
                              BadCounts{"ProductOfBothCountsIsNotTheElementCount", {3, 4}, 3, 5}),
            [](const ::testing::TestParamInfo<BadCounts>& info)
            {
                return std::string(info.param.label);
            });

        TEST_F(Medium_BroadcastSeam, TwoLeadingVectorsOfDifferentCountsAreRefused)
        {
            const Tensor a = filled<float>({3, 4}, quarters<float>(12));
            Tensor c = like(a);
            const Tensor first = filled<float>({3}, quarters<float>(3));
            const Tensor second = filled<float>({12}, quarters<float>(12));
            const std::array<BroadcastOperand, 2> vectors{
                {{&first, BroadcastAxis::Leading}, {&second, BroadcastAxis::Leading}}};

            expect_refusal([&]() { record_sum(a, vectors, c); }, "vector 1");
        }

        TEST_F(Medium_BroadcastSeam, AColumnMajorVectorIsRefusedByName)
        {
            const Tensor a = filled<float>({3, 4}, quarters<float>(12));
            Tensor c = like(a);
            const Tensor v = as_column_major(filled<float>({4}, quarters<float>(4)));
            const std::array<BroadcastOperand, 1> vectors{{{&v, BroadcastAxis::Trailing}}};

            expect_refusal([&]() { record_sum(a, vectors, c); }, "'vector 0' is column-major");
        }

        TEST_F(Medium_BroadcastSeam, AColumnMajorMatrixIsRefusedByName)
        {
            const Tensor a = as_column_major(filled<float>({3, 4}, quarters<float>(12)));
            Tensor c = like(a);
            const Tensor v = filled<float>({4}, quarters<float>(4));
            const std::array<BroadcastOperand, 1> vectors{{{&v, BroadcastAxis::Trailing}}};

            expect_refusal([&]() { record_sum(a, vectors, c); }, "'A' is column-major");
        }

        TEST_F(Medium_BroadcastSeam, AColumnMajorOutputIsRefusedByName)
        {
            const Tensor a = filled<float>({3, 4}, quarters<float>(12));
            Tensor c = as_column_major(like(a));
            const Tensor v = filled<float>({4}, quarters<float>(4));
            const std::array<BroadcastOperand, 1> vectors{{{&v, BroadcastAxis::Trailing}}};

            expect_refusal([&]() { record_sum(a, vectors, c); }, "'C' is column-major");
        }

        TEST_F(Medium_BroadcastSeam, AnOutputThatPartlyOverlapsTheMatrixIsRefused)
        {
            const Tensor arena = engine->create_tensor({8, 3}, Core::DataType::Float32);
            const Tensor a = arena.slice(0, 0, 4);
            Tensor c = arena.slice(0, 2, 6);
            const Tensor v = filled<float>({3}, quarters<float>(3));
            const std::array<BroadcastOperand, 1> vectors{{{&v, BroadcastAxis::Trailing}}};

            expect_refusal([&]() { record_sum(a, vectors, c); }, "partly overlaps 'A'");
        }

        TEST_F(Medium_BroadcastSeam, AnOutputThatOverlapsAVectorIsRefused)
        {
            const Tensor arena = engine->create_tensor({8, 3}, Core::DataType::Float32);
            const Tensor a = arena.slice(0, 0, 4);
            Tensor c = arena.slice(0, 4, 8);
            const Tensor v = arena.reshape({24}).slice(0, 13, 16);
            const std::array<BroadcastOperand, 1> vectors{{{&v, BroadcastAxis::Trailing}}};

            expect_refusal([&]() { record_sum(a, vectors, c); }, "'vector 0' overlaps 'C'");
        }

        TEST_F(Medium_BroadcastSeam, AVectorOfAnotherDtypeThanTheMatrixIsRefused)
        {
            const Tensor a = filled<float>({3, 4}, quarters<float>(12));
            Tensor c = like(a);
            const Tensor v = filled<std::int32_t>({4}, std::vector<std::int32_t>{1, 2, 3, 4});
            const std::array<BroadcastOperand, 1> vectors{{{&v, BroadcastAxis::Trailing}}};

            expect_refusal([&]() { record_sum(a, vectors, c); }, "'vector 0' has dtype");
        }

        TEST_F(Medium_BroadcastSeam, AnOutputOfAnotherDtypeThanTheMatrixIsRefused)
        {
            const Tensor a = filled<float>({3, 4}, quarters<float>(12));
            Tensor c = engine->create_tensor({3, 4}, Core::DataType::Int32);
            const Tensor v = filled<float>({4}, quarters<float>(4));
            const std::array<BroadcastOperand, 1> vectors{{{&v, BroadcastAxis::Trailing}}};

            expect_refusal([&]() { record_sum(a, vectors, c); }, "'C' has dtype");
        }

        TEST_F(Medium_BroadcastSeam, ANullVectorIsRefused)
        {
            const Tensor a = filled<float>({3, 4}, quarters<float>(12));
            Tensor c = like(a);
            const std::array<BroadcastOperand, 1> vectors{{{nullptr, BroadcastAxis::Trailing}}};

            expect_refusal([&]() { record_sum(a, vectors, c); }, "'vector 0' is null");
        }
    } // namespace Test
} // namespace SushiBLAS
