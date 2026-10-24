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
 * @file test_reduction_determinism.cpp
 * @brief Tests that the folding reductions repeat to the bit and sum accurately.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstring>
#include <vector>
#include <cstddef>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/reduction/reduce_1d.hpp>
#include <SushiBLAS/launch/device_policy.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
namespace SushiBLAS
{
    namespace Test
    {
        class Medium_ReductionDeterminism : public SushiBLASTest
        {
            protected:
                /** @brief Below `K_CPU_TWO_STAGE_MIN_ELEMENTS`: the single sequential pass. */
                static constexpr int64_t K_SMALL = 4096;

                /** @brief Above it: the two-stage work-group tree plus an ordered fold. */
                static constexpr int64_t K_LARGE = 1 << 18;

                /** @brief The element count of the var, std and logsumexp case; two-stage too. */
                static constexpr int64_t K_MOMENTS = 100000;

                /** @brief How many times the var, std and logsumexp case runs each reduction. */
                static constexpr int K_MOMENT_RUNS = 10;

                /**
                 * @brief Which reduction shape @p n will actually take on this device.
                 *
                 * Reported in a SCOPED_TRACE so a failure says which of the two kernels
                 * produced it, and asserted on directly by @ref BothShapesAreReachedHere
                 * so that a future retune cannot quietly collapse this file's coverage
                 * onto one path.
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
                 * @brief Runs @p op @p runs times and asserts every result has the same bits.
                 * @param n Element count, which selects the reduction shape.
                 * @see tests/operations/reductions/README.md
                 */
                template<typename Op, typename Gen>
                void expect_bitwise_stable(int64_t n, Op op, Gen gen, int runs = 32)
                {
                    SCOPED_TRACE(is_two_stage(n) ? "two-stage shape" : "single-pass shape");

                    float first = 0.0f;
                    for (int r = 0; r < runs; ++r)
                    {
                        Engine eng(TestEnvironment::runtime());

                        auto t = eng.create_tensor({n});
                        auto res = eng.create_tensor({1});

                        std::vector<float> host(static_cast<std::size_t>(n));
                        for (std::size_t i = 0; i < host.size(); ++i)
                            host[i] = gen(i);
                        TestEnvironment::queue().memcpy(t.data(), host.data(),
                                               host.size() * sizeof(float)).wait();

                        RecordingFixture fixture(eng);
                        op(fixture.recorder(), t, res);
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

                /**
                 * @brief Values for the additive reductions.
                 *
                 * No exact binary representation and a spread of magnitudes, so a changed
                 * combination order lands on different bits rather than cancelling out.
                 */
                static float additive_value(std::size_t i)
                {
                    return 0.1f + static_cast<float>(i % 97) * 0.017f;
                }

                /**
                 * @brief Returns values near 1.0 whose running product stays finite and non-zero.
                 * @see tests/operations/reductions/README.md
                 */
                static float multiplicative_value(std::size_t i)
                {
                    return 1.0f + (static_cast<float>(i % 7) - 3.0f) * 1e-4f;
                }
        };

        /**
         * @brief Tests that the two sizes used here reach both reduction shapes.
         * @see tests/operations/reductions/README.md
         */
        TEST_F(Medium_ReductionDeterminism, BothShapesAreReachedHere)
        {
            auto& q = TestEnvironment::queue();
            const Device::Profile& p = Device::profile_for(q);

            EXPECT_TRUE(is_two_stage(K_LARGE))
                << "kLarge must exercise the two-stage shape.";

            if (p.is_cpu)
            {
                EXPECT_FALSE(is_two_stage(K_SMALL))
                    << "kSmall must exercise the single-pass shape on a CPU.";
            }
        }

        TEST_F(Medium_ReductionDeterminism, SumIsBitwiseStable)
        {
            auto op = [](TaskRecorder& e, const Tensor& t, Tensor& r)
            {
                SushiBLAS::ReductionOps(e).sum(t, r);
            };
            expect_bitwise_stable(K_SMALL, op, additive_value);
            expect_bitwise_stable(K_LARGE, op, additive_value);
        }

        TEST_F(Medium_ReductionDeterminism, MeanIsBitwiseStable)
        {
            auto op = [](TaskRecorder& e, const Tensor& t, Tensor& r)
            {
                SushiBLAS::ReductionOps(e).mean(t, r);
            };
            expect_bitwise_stable(K_SMALL, op, additive_value);
            expect_bitwise_stable(K_LARGE, op, additive_value);
        }

        TEST_F(Medium_ReductionDeterminism, AsumIsBitwiseStable)
        {
            auto op = [](TaskRecorder& e, const Tensor& t, Tensor& r)
            {
                SushiBLAS::ReductionOps(e).asum(t, r);
            };
            expect_bitwise_stable(K_SMALL, op, additive_value);
            expect_bitwise_stable(K_LARGE, op, additive_value);
        }

        TEST_F(Medium_ReductionDeterminism, Nrm2IsBitwiseStable)
        {
            auto op = [](TaskRecorder& e, const Tensor& t, Tensor& r)
            {
                SushiBLAS::ReductionOps(e).nrm2(t, r);
            };
            expect_bitwise_stable(K_SMALL, op, additive_value);
            expect_bitwise_stable(K_LARGE, op, additive_value);
        }

        TEST_F(Medium_ReductionDeterminism, ProdIsBitwiseStable)
        {
            auto op = [](TaskRecorder& e, const Tensor& t, Tensor& r)
            {
                SushiBLAS::ReductionOps(e).prod(t, r);
            };
            expect_bitwise_stable(K_SMALL, op, multiplicative_value);
            expect_bitwise_stable(K_LARGE, op, multiplicative_value);
        }

        /**
         * @brief Tests that var, std and logsumexp repeat to the bit in the two-stage shape.
         */
        TEST_F(Medium_ReductionDeterminism, TwoStageMomentsAreBitwiseStable)
        {
            ASSERT_TRUE(is_two_stage(K_MOMENTS))
                << "kMoments must exercise the two-stage shape.";

            auto var = [](TaskRecorder& e, const Tensor& t, Tensor& r)
            {
                SushiBLAS::ReductionOps(e).var(t, r);
            };
            auto std_dev = [](TaskRecorder& e, const Tensor& t, Tensor& r)
            {
                SushiBLAS::ReductionOps(e).std(t, r);
            };
            auto logsumexp = [](TaskRecorder& e, const Tensor& t, Tensor& r)
            {
                SushiBLAS::ReductionOps(e).logsumexp(t, r);
            };

            expect_bitwise_stable(K_MOMENTS, var, additive_value, K_MOMENT_RUNS);
            expect_bitwise_stable(K_MOMENTS, std_dev, additive_value, K_MOMENT_RUNS);
            expect_bitwise_stable(K_MOMENTS, logsumexp, additive_value, K_MOMENT_RUNS);
        }

        /**
         * @brief Tests that a compensated sum recovers 4096 ones under a head of 1e8.
         * @see tests/operations/reductions/README.md
         */
        TEST_F(Medium_ReductionDeterminism, CompensatedSumRecoversASmallTailUnderALargeHead)
        {
            constexpr int64_t kOnes = 4096;
            constexpr float kHead = 1e8f;

            auto t = engine->create_tensor({kOnes + 1});
            std::vector<float> host(static_cast<std::size_t>(kOnes + 1), 1.0f);
            host[0] = kHead;
            TestEnvironment::queue()
                .memcpy(t.data(), host.data(), host.size() * sizeof(float)).wait();

            auto res = engine->create_tensor({1});
            SushiBLAS::ReductionOps(recorder()).sum(t, res);
            recording->run().value();

            EXPECT_FLOAT_EQ(kHead + static_cast<float>(kOnes), scalar(res))
                << "Naive summation returns " << kHead << " here, because every 1.0 falls "
                   "below the ulp of the running total. Neumaier compensation is what "
                   "recovers the tail.";
        }

        /**
         * @brief Tests that the compensated sum of repeated 0.1f beats naive float summation.
         * @see tests/operations/reductions/README.md
         */
        TEST_F(Medium_ReductionDeterminism, CompensatedSumBeatsNaiveSummationOnRepeatedTenths)
        {
            for (int64_t n : {K_SMALL, K_LARGE})
            {
                SCOPED_TRACE(is_two_stage(n) ? "two-stage shape" : "single-pass shape");

                auto t = filled(n, 0.1f);
                auto res = engine->create_tensor({1});
                SushiBLAS::ReductionOps(recorder()).sum(t, res);
                recording->run().value();

                // 0.1f is 0.100000001490116..., and n copies of it sum to n times that.
                const double exact = static_cast<double>(n) * static_cast<double>(0.1f);

                float naive = 0.0f;
                for (int64_t i = 0; i < n; ++i)
                    naive += 0.1f;

                const double got_err = std::fabs(static_cast<double>(scalar(res)) - exact);
                const double naive_err = std::fabs(static_cast<double>(naive) - exact);

                EXPECT_LT(got_err, naive_err)
                    << "n = " << n << ": the compensated reduction produced "
                    << scalar(res) << " (error " << got_err << ") but naive float "
                    << "summation produces " << naive << " (error " << naive_err
                    << "). Equal errors mean the compensation is not running.";

                // The result is within one ulp of the exact answer at this magnitude.
                EXPECT_LT(got_err, exact * 1e-7)
                    << "n = " << n << ": compensated summation should land within an ulp.";
            }
        }

        /**
         * @brief Tests that the two-stage tree, called directly, agrees with the sequential pass.
         * @see tests/operations/reductions/README.md
         */
        TEST_F(Medium_ReductionDeterminism, TwoStageTreeAgreesWithTheSequentialPass)
        {
            constexpr std::size_t kN = 3000;   // not a multiple of any group width

            auto& q = TestEnvironment::queue();

            auto in = engine->create_tensor({static_cast<int64_t>(kN)});
            auto out_seq = engine->create_tensor({1});
            auto out_tree = engine->create_tensor({1});

            std::vector<float> host(kN);
            for (std::size_t i = 0; i < kN; ++i)
                host[i] = 0.1f + static_cast<float>(i % 31) * 0.031f;
            q.memcpy(in.data(), host.data(), host.size() * sizeof(float)).wait();

            // Scratch enough for the widest tree Reduce::plan can ask for.
            auto scratch = engine->create_tensor(
                {static_cast<int64_t>(Reduce::K_MAX_GROUPS)});

            const auto identity = [](float x) { return x; };
            const auto keep = [](float v, [[maybe_unused]] std::size_t n) { return v; };

            Reduce::reduce<Reduce::CompensatedSum>(
                q, in.data_as<float>(), out_seq.data_as<float>(), kN,
                static_cast<float*>(nullptr), 0, identity, keep, {}).wait();

            Reduce::reduce<Reduce::CompensatedSum>(
                q, in.data_as<float>(), out_tree.data_as<float>(), kN,
                scratch.data_as<float>(), Reduce::K_MAX_GROUPS, identity, keep, {}).wait();

            const double exact = [&]
            {
                double s = 0.0;
                for (float v : host)
                    s += static_cast<double>(v);
                return s;
            }();

            // The two shapes sum in different orders. See tests/operations/reductions/README.md.
            EXPECT_NEAR(static_cast<double>(scalar(out_tree)), exact, exact * 1e-6);
            EXPECT_NEAR(static_cast<double>(scalar(out_seq)), exact, exact * 1e-6);
        }

    } // namespace Test
} // namespace SushiBLAS
