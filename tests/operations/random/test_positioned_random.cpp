/**************************************************************************/
/* test_positioned_random.cpp                                             */
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
 * @file test_positioned_random.cpp
 * @brief Tests that PositionedRandomOps::bernoulli draws RandomOps::bernoulli's bits at the position it reads.
 * @author Mustafa Garip
 */

#include <vector>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <gtest/gtest.h>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/random/random.hpp>
#include <SushiBLAS/random/positioned_random.hpp>
#include <SushiBLAS/random/rng_position_tensor.hpp>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/engine/execution_mode.hpp>
#include <SushiBLAS/engine/late_count_hold.hpp>
#include <SushiBLAS/engine/runtime_scratch_source.hpp>
#include <SushiBLAS/engine/scratch_pool.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/scalar_dtype.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief Holds the element count of an equality case: 4 * 1024 + 3, a partial last block.
            constexpr int64_t K_ELEMENTS = 4099;

            /// @brief Holds the element count of the mean case: 2^20.
            constexpr int64_t K_MEAN_ELEMENTS = int64_t{1} << 20;

            /// @brief Holds the probability the cases draw at.
            constexpr double K_P = 0.3;

            /// @brief Holds the first stream position the cases draw at.
            constexpr RNGPosition K_POSITION_A{20261116, 5};

            /// @brief Holds a second position, one seed and one offset away from K_POSITION_A.
            constexpr RNGPosition K_POSITION_B{20261117, 9};

            /**
             * @brief Holds how far the mean of K_MEAN_ELEMENTS draws may sit from K_P.
             *
             * The mean of n Bernoulli(p) draws has standard deviation sqrt(p (1 - p) / n), which is
             * 4.5e-4 for p of 0.3 and n of 2^20. The bound is 11 of them, so a correct draw fails it
             * with probability below 1e-27; a fixed seed makes the case repeat.
             */
            constexpr double K_MEAN_BOUND = 0.005;
        } // namespace

        /** @brief Builds tensors and draws on the fixture's engine. */
        class Medium_PositionedBernoulli : public SushiBLASTest
        {
            protected:
                /** @brief Returns a Shared INT64 tensor holding position. */
                [[nodiscard]] Tensor position_tensor(RNGPosition position)
                {
                    Tensor tensor = engine->create_tensor({K_RNG_POSITION_ELEMENTS},
                                                          Core::DataType::Int64);
                    write_rng_position(tensor, position);
                    return tensor;
                }

                /** @brief Returns the n elements of a tensor of dtype T that PositionedRandomOps leaves at position. */
                template<typename T>
                [[nodiscard]] std::vector<T> draw_positioned(RNGPosition position, int64_t n,
                                                             double p)
                {
                    Tensor holder = position_tensor(position);
                    Tensor t = engine->create_tensor({n}, sushiblas_dtype<T>());
                    PositionedRandomOps(recorder()).bernoulli(t, p, holder);
                    EXPECT_TRUE(recording->run_succeeds());
                    return std::vector<T>(t.data_as<T>(), t.data_as<T>() + n);
                }

                /** @brief Returns the n elements of a tensor of dtype T that RandomOps leaves at position. */
                template<typename T>
                [[nodiscard]] std::vector<T> draw_plain(RNGPosition position, int64_t n, double p)
                {
                    engine->rng().reseed(position.seed);
                    static_cast<void>(engine->rng().reserve(position.offset));
                    Tensor t = engine->create_tensor({n}, sushiblas_dtype<T>());
                    RandomOps(recorder(), engine->rng()).bernoulli(t, p);
                    EXPECT_TRUE(recording->run_succeeds());
                    return std::vector<T>(t.data_as<T>(), t.data_as<T>() + n);
                }

                /**
                 * @brief Records one positioned fill, executes its plan once per entry of positions
                 *        and returns the n elements each execution left.
                 * @param positions The position the host writes before each execution.
                 */
                [[nodiscard]] std::vector<std::vector<float>> replay_at(
                    const std::vector<RNGPosition>& positions, int64_t n, double p)
                {
                    SushiRuntime::Runtime& runtime = TestEnvironment::runtime();
                    SushiRuntime::Graph graph = runtime.graph();
                    RuntimeScratchSource scratch_source(runtime);
                    ScratchPool scratch_pool(scratch_source);
                    LateCountHold late_counts(runtime);
                    TaskRecorder replay_recorder(graph.recorder(), scratch_pool, late_counts);
                    Tensor holder = position_tensor(positions.front());
                    Tensor t = engine->create_tensor({n}, Core::DataType::Float32);
                    PositionedRandomOps(replay_recorder).bernoulli(t, p, holder);

                    std::vector<std::vector<float>> masks;
                    for (const RNGPosition& position : positions)
                    {
                        write_rng_position(holder, position);
                        SushiRuntime::RunReport report;
                        EXPECT_TRUE(graph.run(report));
                        masks.emplace_back(t.data_as<float>(), t.data_as<float>() + n);
                    }
                    graph.clear();
                    late_counts.release();
                    return masks;
                }

                /** @brief Returns the n elements of a FLOAT32 tensor drawn at position under mode. */
                [[nodiscard]] std::vector<float> draw_under(ExecutionMode mode,
                                                            RNGPosition position, int64_t n)
                {
                    set_mode(mode);
                    return draw_positioned<float>(position, n, K_P);
                }
        };

        /** @brief Tests that a FLOAT32 fill equals RandomOps::bernoulli at the same stream position, bit for bit. */
        TEST_F(Medium_PositionedBernoulli, Float32EqualsRandomOpsBernoulliAtOnePosition)
        {
            const std::vector<float> plain = draw_plain<float>(K_POSITION_A, K_ELEMENTS, K_P);
            const std::vector<float> positioned =
                draw_positioned<float>(K_POSITION_A, K_ELEMENTS, K_P);

            EXPECT_EQ(positioned, plain);
        }

        /** @brief Tests that a FLOAT64 fill equals RandomOps::bernoulli at the same stream position, bit for bit. */
        TEST_F(Medium_PositionedBernoulli, Float64EqualsRandomOpsBernoulliAtOnePosition)
        {
            const std::vector<double> plain = draw_plain<double>(K_POSITION_A, K_ELEMENTS, K_P);
            const std::vector<double> positioned =
                draw_positioned<double>(K_POSITION_A, K_ELEMENTS, K_P);

            EXPECT_EQ(positioned, plain);
        }

        /** @brief Tests that positions A, B, A give masks m1, m2, m1 from one plan compiled once, with m1 not m2. */
        TEST_F(Medium_PositionedBernoulli, OnePlanExecutedAtPositionsABAGivesMasksM1M2M1)
        {
            const std::vector<std::vector<float>> masks =
                replay_at({K_POSITION_A, K_POSITION_B, K_POSITION_A}, K_ELEMENTS, K_P);

            ASSERT_EQ(masks.size(), std::size_t{3});
            EXPECT_EQ(masks[2], masks[0]);
            EXPECT_NE(masks[1], masks[0]);
        }

        /** @brief Tests that each execution of a plan equals the plain fill at the position written before it. */
        TEST_F(Medium_PositionedBernoulli, EachExecutionOfAPlanEqualsThePlainFillAtItsPosition)
        {
            const std::vector<std::vector<float>> masks =
                replay_at({K_POSITION_B, K_POSITION_A}, K_ELEMENTS, K_P);
            const std::vector<float> plain_b = draw_plain<float>(K_POSITION_B, K_ELEMENTS, K_P);
            const std::vector<float> plain_a = draw_plain<float>(K_POSITION_A, K_ELEMENTS, K_P);

            ASSERT_EQ(masks.size(), std::size_t{2});
            EXPECT_EQ(masks[0], plain_b);
            EXPECT_EQ(masks[1], plain_a);
        }

        /** @brief Tests that a position in the second half of a larger allocation is the one the fill reads. */
        TEST_F(Medium_PositionedBernoulli, ReadsAPositionThatIsAViewIntoALargerAllocation)
        {
            Tensor larger = engine->create_tensor({4}, Core::DataType::Int64);
            Tensor view = larger.slice(0, 2, 4);
            write_rng_position(view, K_POSITION_A);
            Tensor t = engine->create_tensor({K_ELEMENTS}, Core::DataType::Float32);

            PositionedRandomOps(recorder()).bernoulli(t, K_P, view);
            ASSERT_TRUE(recording->run_succeeds());

            EXPECT_EQ(std::vector<float>(t.data_as<float>(), t.data_as<float>() + K_ELEMENTS),
                      draw_plain<float>(K_POSITION_A, K_ELEMENTS, K_P));
        }

        /** @brief Tests that the mean of 2^20 draws sits within K_MEAN_BOUND of p. */
        TEST_F(Medium_PositionedBernoulli, TheMeanOfManyDrawsIsNearP)
        {
            const std::vector<float> draws =
                draw_positioned<float>(K_POSITION_A, K_MEAN_ELEMENTS, K_P);
            const double mean = std::accumulate(draws.begin(), draws.end(), 0.0) /
                                static_cast<double>(K_MEAN_ELEMENTS);

            EXPECT_NEAR(mean, K_P, K_MEAN_BOUND);
        }

        /** @brief Tests that ForceReference writes the bits the device writes. */
        TEST_F(Medium_PositionedBernoulli, ForceReferenceEqualsTheDeviceBitForBit)
        {
            const std::vector<float> reference =
                draw_under(ExecutionMode::ForceReference, K_POSITION_A, K_ELEMENTS);
            const std::vector<float> device =
                draw_under(ExecutionMode::ForceSYCL, K_POSITION_A, K_ELEMENTS);

            EXPECT_EQ(device, reference);
        }

        /** @brief Tests that p of 0 writes only zeros. */
        TEST_F(Medium_PositionedBernoulli, AProbabilityOfZeroWritesOnlyZeros)
        {
            const std::vector<float> draws = draw_positioned<float>(K_POSITION_A, K_ELEMENTS, 0.0);

            EXPECT_EQ(draws, std::vector<float>(K_ELEMENTS, 0.0f));
        }

        /** @brief Tests that p of 1 writes only ones. */
        TEST_F(Medium_PositionedBernoulli, AProbabilityOfOneWritesOnlyOnes)
        {
            const std::vector<float> draws = draw_positioned<float>(K_POSITION_A, K_ELEMENTS, 1.0);

            EXPECT_EQ(draws, std::vector<float>(K_ELEMENTS, 1.0f));
        }

        /** @brief Tests that a HALF tensor is refused and nothing is recorded. */
        TEST_F(Medium_PositionedBernoulli, RefusesHalfAndRecordsNothing)
        {
            Tensor holder = position_tensor(K_POSITION_A);
            Tensor t = engine->create_tensor({K_ELEMENTS}, Core::DataType::Half);

            EXPECT_THROW(PositionedRandomOps(recorder()).bernoulli(t, K_P, holder),
                         std::invalid_argument);
            EXPECT_EQ(recording->recorded_count(), std::size_t{0});
        }

        /** @brief Tests that a FLOAT32 position is refused and nothing is recorded. */
        TEST_F(Medium_PositionedBernoulli, RefusesAFloat32PositionAndRecordsNothing)
        {
            Tensor holder = engine->create_tensor({K_RNG_POSITION_ELEMENTS},
                                                  Core::DataType::Float32);
            Tensor t = engine->create_tensor({K_ELEMENTS}, Core::DataType::Float32);

            EXPECT_THROW(PositionedRandomOps(recorder()).bernoulli(t, K_P, holder),
                         std::invalid_argument);
            EXPECT_EQ(recording->recorded_count(), std::size_t{0});
        }

        /** @brief Tests that a position of three elements is refused and nothing is recorded. */
        TEST_F(Medium_PositionedBernoulli, RefusesAPositionOfThreeElementsAndRecordsNothing)
        {
            Tensor holder = engine->create_tensor({3}, Core::DataType::Int64);
            Tensor t = engine->create_tensor({K_ELEMENTS}, Core::DataType::Float32);

            EXPECT_THROW(PositionedRandomOps(recorder()).bernoulli(t, K_P, holder),
                         std::invalid_argument);
            EXPECT_EQ(recording->recorded_count(), std::size_t{0});
        }

        /** @brief Tests that a position that shares bytes with t is refused and nothing is recorded. */
        TEST_F(Medium_PositionedBernoulli, RefusesAPositionThatOverlapsTAndRecordsNothing)
        {
            Tensor holder = position_tensor(K_POSITION_A);
            const std::vector<int64_t> dims{4};
            Tensor t(holder.storage, SushiRuntime::span<const int64_t>(dims.data(), dims.size()),
                     Core::DataType::Float32, 0, engine->layout());

            EXPECT_THROW(PositionedRandomOps(recorder()).bernoulli(t, K_P, holder),
                         std::invalid_argument);
            EXPECT_EQ(recording->recorded_count(), std::size_t{0});
        }

        /** @brief Tests that a p below 0, above 1 or NaN is refused and nothing is recorded. */
        TEST_F(Medium_PositionedBernoulli, RefusesAProbabilityOutsideZeroToOneAndRecordsNothing)
        {
            Tensor holder = position_tensor(K_POSITION_A);
            Tensor t = engine->create_tensor({K_ELEMENTS}, Core::DataType::Float32);
            PositionedRandomOps random(recorder());

            EXPECT_THROW(random.bernoulli(t, -0.1, holder), std::invalid_argument);
            EXPECT_THROW(random.bernoulli(t, 1.5, holder), std::invalid_argument);
            EXPECT_THROW(random.bernoulli(t, std::numeric_limits<double>::quiet_NaN(), holder),
                         std::invalid_argument);
            EXPECT_EQ(recording->recorded_count(), std::size_t{0});
        }
    } // namespace Test
} // namespace SushiBLAS
