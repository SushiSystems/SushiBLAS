/**************************************************************************/
/* test_reduction_determinism.cpp                                         */
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
 * @file test_reduction_determinism.cpp
 * @brief Tests that the folding reductions repeat to the bit and sum accurately.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/reduction/reduce_1d.hpp>
#include <SushiBLAS/launch/device_policy.hpp>
#include "support/tolerance.hpp"
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief Below `K_CPU_TWO_STAGE_MIN_ELEMENTS`: the single sequential pass.
            constexpr int64_t K_SMALL = 4096;

            /// @brief Above it: the two-stage work-group tree plus an ordered fold.
            constexpr int64_t K_LARGE = 1 << 18;

            /// @brief The element count of the var, std and logsumexp case; two-stage too.
            constexpr int64_t K_MOMENTS = 100000;

            /// @brief How many times a sum-like reduction runs in a stability case.
            constexpr int K_RUNS = 32;

            /// @brief How many times the var, std and logsumexp case runs each reduction.
            constexpr int K_MOMENT_RUNS = 10;

            /// @brief The rounded operations on the longest path of a compensated sum, as one ulp.
            constexpr std::size_t K_COMPENSATED_ROUNDINGS = 2;

            /// @brief The element count of the direct two-stage case; no group width divides it.
            constexpr std::size_t K_TREE_COUNT = 3000;

            /// @brief The ones that follow the head of the compensated-tail input.
            constexpr int64_t K_ONES = 4096;

            /// @brief The head of the compensated-tail input; its ulp in float is 8.
            constexpr float K_HEAD = 1e8f;

            /// @brief The unit CPU profile of the planner cases.
            Device::Profile cpu_profile()
            {
                Device::Profile profile;
                profile.is_cpu = true;
                profile.compute_units = 8;
                profile.max_work_group_size = 8192;
                return profile;
            }

            /**
             * @brief Values for the additive reductions.
             *
             * No exact binary representation and a spread of magnitudes, so a changed
             * combination order lands on different bits rather than cancelling out.
             */
            float additive_value(std::size_t i)
            {
                return 0.1f + static_cast<float>(i % 97) * 0.017f;
            }

            /**
             * @brief Returns values near 1.0 whose running product stays finite and non-zero.
             * @see tests/operations/reductions/README.md
             */
            float multiplicative_value(std::size_t i)
            {
                return 1.0f + (static_cast<float>(i % 7) - 3.0f) * 1e-4f;
            }

            /** @brief Returns the sum of n copies of 0.1f added one after the other in float. */
            float naive_sum_of_tenths(int64_t n)
            {
                float naive = 0.0f;
                for (int64_t i = 0; i < n; ++i)
                    naive += 0.1f;
                return naive;
            }

            /** @brief Returns kN positive values that differ from one another. */
            std::vector<float> graded_values()
            {
                std::vector<float> host(K_TREE_COUNT);
                for (std::size_t i = 0; i < host.size(); ++i)
                    host[i] = 0.1f + static_cast<float>(i % 31) * 0.031f;
                return host;
            }

            /** @brief Returns the levels of the halving tree that folds the partials of stage two. */
            std::size_t tree_levels(std::size_t partials)
            {
                std::size_t levels = 0;
                for (std::size_t width = 1; width < partials; width *= 2)
                    ++levels;
                return levels;
            }

            /** @brief Returns one large head followed by K_ONES ones. */
            std::vector<float> head_then_ones()
            {
                std::vector<float> host(static_cast<std::size_t>(K_ONES + 1), 1.0f);
                host[0] = K_HEAD;
                return host;
            }

            /** @brief The reduction a stability row records. */
            using RecordOp = void (*)(TaskRecorder&, const Tensor&, Tensor&);

            /** @brief The generator of the input a stability row reduces. */
            using Generator = float (*)(std::size_t);
        }

        /** @brief One reduction, its input generator and the size and run count it is held to. */
        struct StabilityRow
        {
            std::string name;
            RecordOp op;
            Generator gen;
            int64_t n;
            int runs;
        };

        /** @brief Returns the folding reductions at a size on each side of the CPU threshold. */
        static std::vector<StabilityRow> stability_rows()
        {
            const RecordOp sum = [](TaskRecorder& e, const Tensor& t, Tensor& r)
            { SushiBLAS::ReductionOps(e).sum(t, r); };
            const RecordOp mean = [](TaskRecorder& e, const Tensor& t, Tensor& r)
            { SushiBLAS::ReductionOps(e).mean(t, r); };
            const RecordOp asum = [](TaskRecorder& e, const Tensor& t, Tensor& r)
            { SushiBLAS::ReductionOps(e).asum(t, r); };
            const RecordOp nrm2 = [](TaskRecorder& e, const Tensor& t, Tensor& r)
            { SushiBLAS::ReductionOps(e).nrm2(t, r); };
            const RecordOp prod = [](TaskRecorder& e, const Tensor& t, Tensor& r)
            { SushiBLAS::ReductionOps(e).prod(t, r); };
            const RecordOp var = [](TaskRecorder& e, const Tensor& t, Tensor& r)
            { SushiBLAS::ReductionOps(e).var(t, r); };
            const RecordOp std_dev = [](TaskRecorder& e, const Tensor& t, Tensor& r)
            { SushiBLAS::ReductionOps(e).std(t, r); };
            const RecordOp logsumexp = [](TaskRecorder& e, const Tensor& t, Tensor& r)
            { SushiBLAS::ReductionOps(e).logsumexp(t, r); };

            return {
                {"SumAtTheSinglePassSize", sum, additive_value, K_SMALL, K_RUNS},
                {"SumAtTheTwoStageSize", sum, additive_value, K_LARGE, K_RUNS},
                {"MeanAtTheSinglePassSize", mean, additive_value, K_SMALL, K_RUNS},
                {"MeanAtTheTwoStageSize", mean, additive_value, K_LARGE, K_RUNS},
                {"AsumAtTheSinglePassSize", asum, additive_value, K_SMALL, K_RUNS},
                {"AsumAtTheTwoStageSize", asum, additive_value, K_LARGE, K_RUNS},
                {"Nrm2AtTheSinglePassSize", nrm2, additive_value, K_SMALL, K_RUNS},
                {"Nrm2AtTheTwoStageSize", nrm2, additive_value, K_LARGE, K_RUNS},
                {"ProdAtTheSinglePassSize", prod, multiplicative_value, K_SMALL, K_RUNS},
                {"ProdAtTheTwoStageSize", prod, multiplicative_value, K_LARGE, K_RUNS},
                {"VarAtTheMomentsSize", var, additive_value, K_MOMENTS, K_MOMENT_RUNS},
                {"StdAtTheMomentsSize", std_dev, additive_value, K_MOMENTS, K_MOMENT_RUNS},
                {"LogSumExpAtTheMomentsSize", logsumexp, additive_value, K_MOMENTS,
                 K_MOMENT_RUNS},
            };
        }

        /** @brief Names a row by its description, which gtest takes as the test's suffix. */
        template<typename Row>
        static std::string row_name(const ::testing::TestParamInfo<Row>& info)
        {
            return info.param.name;
        }

        class Medium_ReductionDeterminism : public SushiBLASTest
        {
            protected:
                /**
                 * @brief Which reduction shape @p n will actually take on this device.
                 *
                 * Reported in a SCOPED_TRACE so a failure says which of the two kernels
                 * produced it, and asserted on directly by the shape cases so that a future
                 * retune cannot quietly collapse this file's coverage onto one path.
                 */
                [[nodiscard]] static bool is_two_stage(int64_t n)
                {
                    auto& q = TestEnvironment::queue();
                    return Reduce::planned_partials(Device::profile_for(q),
                                                    static_cast<std::size_t>(n)) > 0;
                }

                /** @brief A tensor of @p n copies of @p v. */
                Tensor filled(int64_t n, float v)
                {
                    auto t = engine->create_tensor({n});
                    const std::vector<float> host(static_cast<std::size_t>(n), v);
                    TestEnvironment::queue()
                        .memcpy(t.data(), host.data(), host.size() * sizeof(float)).wait();
                    return t;
                }

                /** @brief The scalar in @p t, copied back to the host. */
                [[nodiscard]] static float scalar(const Tensor& t)
                {
                    float out = 0.0f;
                    TestEnvironment::queue()
                        .memcpy(&out, t.data(), sizeof(float)).wait();
                    return out;
                }

                /**
                 * @brief Runs a row's reduction its run count and asserts every result has the same bits.
                 * @see tests/operations/reductions/README.md
                 */
                void expect_bitwise_stable(const StabilityRow& row)
                {
                    SCOPED_TRACE(is_two_stage(row.n) ? "two-stage shape" : "single-pass shape");

                    float first = 0.0f;
                    for (int r = 0; r < row.runs; ++r)
                    {
                        Engine eng(TestEnvironment::runtime());

                        auto t = eng.create_tensor({row.n});
                        auto res = eng.create_tensor({1});

                        std::vector<float> host(static_cast<std::size_t>(row.n));
                        for (std::size_t i = 0; i < host.size(); ++i)
                            host[i] = row.gen(i);
                        TestEnvironment::queue().memcpy(t.data(), host.data(),
                                               host.size() * sizeof(float)).wait();

                        RecordingFixture fixture(TestEnvironment::runtime(), eng);
                        row.op(fixture.recorder(), t, res);
                        fixture.run().value();

                        const float got = scalar(res);
                        if (r == 0)
                        {
                            // A degenerate result is vacuous. See the test README.
                            ASSERT_TRUE(std::isfinite(got) && got != 0.0f)
                                << "the generated input reduced to " << got
                                << ", which is bit-stable for reasons that have nothing "
                                   "to do with combination order.";
                            first = got;
                            continue;
                        }

                        EXPECT_EQ(0, std::memcmp(&first, &got, sizeof(float)))
                            << "run " << r << " produced " << got << " where run 0 produced "
                            << first << ". A reduction must be reproducible to the bit, not "
                               "merely to a tolerance.";
                    }
                }

                /** @brief Returns the sum of n copies of 0.1f the device produces. */
                float compensated_sum_of_tenths(int64_t n)
                {
                    SCOPED_TRACE(is_two_stage(n) ? "two-stage shape" : "single-pass shape");

                    auto t = filled(n, 0.1f);
                    auto res = engine->create_tensor({1});
                    SushiBLAS::ReductionOps(recorder()).sum(t, res);
                    recording->run().value();
                    return scalar(res);
                }
        };

        /**
         * @brief Tests that a CPU takes the single sequential pass at the small size.
         * @see tests/operations/reductions/README.md
         */
        TEST(Small_ReductionDeterminism, ACPUTakesTheSinglePassAtTheSmallSize)
        {
            EXPECT_EQ(Reduce::planned_partials(cpu_profile(), static_cast<std::size_t>(K_SMALL)),
                      std::size_t{0});
        }

        /**
         * @brief Tests that a CPU takes the two-stage tree at the large size.
         * @see tests/operations/reductions/README.md
         */
        TEST(Small_ReductionDeterminism, ACPUTakesTheTwoStageTreeAtTheLargeSize)
        {
            EXPECT_GT(Reduce::planned_partials(cpu_profile(), static_cast<std::size_t>(K_LARGE)),
                      std::size_t{0});
        }

        /**
         * @brief Tests that the large size reaches the two-stage shape on the test device.
         * @see tests/operations/reductions/README.md
         */
        TEST_F(Medium_ReductionDeterminism, TheLargeSizeReachesTheTwoStageShapeHere)
        {
            EXPECT_TRUE(is_two_stage(K_LARGE)) << "kLarge must exercise the two-stage shape.";
        }

        /**
         * @brief Tests that the moments size reaches the two-stage shape on the test device.
         * @see tests/operations/reductions/README.md
         */
        TEST_F(Medium_ReductionDeterminism, TheMomentsSizeReachesTheTwoStageShapeHere)
        {
            EXPECT_TRUE(is_two_stage(K_MOMENTS)) << "kMoments must exercise the two-stage shape.";
        }

        /** @brief Runs one reduction at one size for its run count. */
        class Medium_ReductionStability : public Medium_ReductionDeterminism,
                                          public ::testing::WithParamInterface<StabilityRow>
        {
        };

        /**
         * @brief Tests that a reduction repeats to the bit at its size.
         * @see tests/operations/reductions/README.md
         */
        TEST_P(Medium_ReductionStability, TheReductionIsBitwiseStable)
        {
            expect_bitwise_stable(GetParam());
        }

        INSTANTIATE_TEST_SUITE_P(, Medium_ReductionStability, ::testing::ValuesIn(stability_rows()),
                                 row_name<StabilityRow>);

        /**
         * @brief Tests that a compensated sum recovers 4096 ones under a head of 1e8.
         *
         * Naive summation returns 1e8, an error of 4096. The bound counts the roundings of a
         * compensated sum plus one per level of the tree that adds the stage-two partials plainly.
         * @see tests/operations/reductions/README.md
         */
        TEST_F(Medium_ReductionDeterminism, CompensatedSumRecoversASmallTailUnderALargeHead)
        {
            const std::vector<float> host = head_then_ones();
            auto t = engine->create_tensor({K_ONES + 1});
            TestEnvironment::queue()
                .memcpy(t.data(), host.data(), host.size() * sizeof(float)).wait();

            auto res = engine->create_tensor({1});
            SushiBLAS::ReductionOps(recorder()).sum(t, res);
            recording->run().value();

            const std::size_t partials = Reduce::planned_partials(
                Device::profile_for(TestEnvironment::queue()),
                static_cast<std::size_t>(K_ONES + 1));
            const double exact = static_cast<double>(K_HEAD) + static_cast<double>(K_ONES);

            EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_relative_error<float>(
                static_cast<double>(scalar(res)), exact,
                K_COMPENSATED_ROUNDINGS + tree_levels(partials))));
        }

        /** @brief Sums n copies of 0.1f at one size on each side of the CPU threshold. */
        class Medium_ReductionTenths : public Medium_ReductionDeterminism,
                                       public ::testing::WithParamInterface<int64_t>
        {
        };

        /**
         * @brief Tests that the compensated sum of repeated 0.1f is nearer the exact sum than naive float summation.
         * @see tests/operations/reductions/README.md
         */
        TEST_P(Medium_ReductionTenths, CompensatedSumBeatsNaiveSummation)
        {
            const int64_t n = GetParam();
            // 0.1f is 0.100000001490116..., and n copies of it sum to n times that.
            const double exact = static_cast<double>(n) * static_cast<double>(0.1f);

            const float got = compensated_sum_of_tenths(n);
            const float naive = naive_sum_of_tenths(n);

            EXPECT_LT(std::fabs(static_cast<double>(got) - exact),
                      std::fabs(static_cast<double>(naive) - exact))
                << "n = " << n << ": the compensated reduction produced " << got
                << " but naive float summation produces " << naive
                << ". Equal errors mean the compensation is not running.";
        }

        /**
         * @brief Tests that the compensated sum of repeated 0.1f lands within the rounding bound of the exact sum.
         * @see tests/operations/reductions/README.md
         */
        TEST_P(Medium_ReductionTenths, CompensatedSumLandsWithinTheRoundingBound)
        {
            const int64_t n = GetParam();
            const double exact = static_cast<double>(n) * static_cast<double>(0.1f);

            EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_relative_error<float>(
                static_cast<double>(compensated_sum_of_tenths(n)), exact,
                K_COMPENSATED_ROUNDINGS)));
        }

        INSTANTIATE_TEST_SUITE_P(, Medium_ReductionTenths, ::testing::Values(K_SMALL, K_LARGE));

        /**
         * @brief Tests that the two-stage tree, called directly, is within the rounding bound of the exact sum.
         * @see tests/operations/reductions/README.md
         */
        TEST_F(Medium_ReductionDeterminism, TheTwoStageTreeSumsWithinTheRoundingBound)
        {
            auto& q = TestEnvironment::queue();
            const std::vector<float> host = graded_values();
            auto in = engine->create_tensor({static_cast<int64_t>(K_TREE_COUNT)});
            auto out_tree = engine->create_tensor({1});
            q.memcpy(in.data(), host.data(), host.size() * sizeof(float)).wait();

            // Scratch enough for the widest tree Reduce::plan can ask for.
            auto scratch = engine->create_tensor({static_cast<int64_t>(Reduce::K_MAX_GROUPS)});

            const auto identity = [](float x) { return x; };
            const auto keep = [](float v, [[maybe_unused]] std::size_t n) { return v; };

            Reduce::reduce<Reduce::CompensatedSum>(
                q, in.data_as<float>(), out_tree.data_as<float>(), K_TREE_COUNT,
                scratch.data_as<float>(), Reduce::K_MAX_GROUPS, identity, keep, {}).wait();

            // The values are positive, so the sum is its own magnitude.
            const double exact = std::accumulate(host.begin(), host.end(), 0.0);

            // The two shapes sum in different orders. See tests/operations/reductions/README.md.
            EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_sum_error<float>(
                static_cast<double>(scalar(out_tree)), exact, exact, K_TREE_COUNT)));
        }

        /**
         * @brief Tests that the sequential pass, called directly, is within the rounding bound of the exact sum.
         * @see tests/operations/reductions/README.md
         */
        TEST_F(Medium_ReductionDeterminism, TheSequentialPassSumsWithinTheRoundingBound)
        {
            auto& q = TestEnvironment::queue();
            const std::vector<float> host = graded_values();
            auto in = engine->create_tensor({static_cast<int64_t>(K_TREE_COUNT)});
            auto out_seq = engine->create_tensor({1});
            q.memcpy(in.data(), host.data(), host.size() * sizeof(float)).wait();

            const auto identity = [](float x) { return x; };
            const auto keep = [](float v, [[maybe_unused]] std::size_t n) { return v; };

            Reduce::reduce<Reduce::CompensatedSum>(
                q, in.data_as<float>(), out_seq.data_as<float>(), K_TREE_COUNT,
                static_cast<float*>(nullptr), 0, identity, keep, {}).wait();

            const double exact = std::accumulate(host.begin(), host.end(), 0.0);

            EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_sum_error<float>(
                static_cast<double>(scalar(out_seq)), exact, exact, K_TREE_COUNT)));
        }

    } // namespace Test
} // namespace SushiBLAS
