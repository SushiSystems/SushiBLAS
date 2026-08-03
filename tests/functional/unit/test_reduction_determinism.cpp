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
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

// The two properties the folding reductions were rebuilt for: a bit-identical
// answer on every run, and a summation accurate enough to be worth trusting.
//
// Both are asserted at BOTH reduction shapes. `Reduce::planned_partials` sends a
// CPU below 16K elements down a single sequential pass and everything else down
// the two-stage work-group tree, so a suite that only ever reduced 100 elements
// would leave the entire parallel path - the one a GPU always takes, and the one
// where combination order is actually in question - untested. Every case below
// therefore runs at a size on each side of that threshold.
//
// The *segmented* reductions - `max_along` / `sum_along`, whose answer is a vector
// rather than a scalar - assert the same two properties in
// test_axis_reductions.cpp, at both of their own launch shapes. They live there
// rather than here because every helper in this file is built around reading one
// float back.

#include <cmath>
#include <cstring>
#include <vector>
#include <cstddef>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include <SushiBLAS/engine/reduce_1d.hpp>
#include <SushiBLAS/engine/device_policy.hpp>
#include "../../common/test_utils.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        class Unit_ReductionDeterminism : public SushiBLASTest
        {
            protected:
                /** @brief Below `K_CPU_TWO_STAGE_MIN_ELEMENTS`: the single sequential pass. */
                static constexpr int64_t kSmall = 4096;

                /** @brief Above it: the two-stage work-group tree plus an ordered fold. */
                static constexpr int64_t kLarge = 1 << 18;

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
                    auto& q = TestEnvironment::get_context().get_queue();
                    return Reduce::planned_partials(Device::profile_for(q),
                                                    static_cast<std::size_t>(n)) > 0;
                }

                /** @brief A tensor of @p n copies of @p v. */
                Tensor filled(int64_t n, float v)
                {
                    auto t = engine->create_tensor({n});
                    const std::vector<float> host(static_cast<std::size_t>(n), v);
                    TestEnvironment::get_context().get_queue()
                        .memcpy(t.data(), host.data(), host.size() * sizeof(float)).wait();
                    return t;
                }

                /** @brief The scalar in @p t, copied back to the host. */
                [[nodiscard]] static float scalar(const Tensor& t)
                {
                    float out = 0.0f;
                    TestEnvironment::get_context().get_queue()
                        .memcpy(&out, t.data(), sizeof(float)).wait();
                    return out;
                }

                /**
                 * @brief Runs @p op @p runs times and asserts every result is the same bits.
                 *
                 * `std::memcmp`, not `EXPECT_NEAR`. The guarantee SushiBLAS inherits from
                 * `sushiruntime_deterministic_fp` is byte equality for the same binary on
                 * the same device, and a tolerance-based check cannot tell the difference
                 * between that and "close enough", which is the whole distinction under
                 * test. A fresh Engine per run so nothing is reused between them.
                 *
                 * @param n Element count, which selects the reduction shape.
                 * @param op Records one reduction on the given engine.
                 * @param runs How many times to repeat it.
                 */
                template<typename Op, typename Gen>
                void expect_bitwise_stable(int64_t n, Op op, Gen gen, int runs = 32)
                {
                    SCOPED_TRACE(is_two_stage(n) ? "two-stage shape" : "single-pass shape");

                    float first = 0.0f;
                    for (int r = 0; r < runs; ++r)
                    {
                        auto& ctx = TestEnvironment::get_context();
                        Engine eng(ctx);

                        auto t = eng.create_tensor({n});
                        auto res = eng.create_tensor({1});

                        std::vector<float> host(static_cast<std::size_t>(n));
                        for (std::size_t i = 0; i < host.size(); ++i)
                            host[i] = gen(i);
                        ctx.get_queue().memcpy(t.data(), host.data(),
                                               host.size() * sizeof(float)).wait();

                        op(eng, t, res);
                        eng.execute().wait();

                        const float got = scalar(res);
                        if (r == 0)
                        {
                            // A degenerate result would make byte equality trivial and
                            // this whole case vacuous - zero, an infinity or a NaN
                            // compares stable no matter how the reduction combined. So
                            // the input has to actually produce a number first.
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
                 * @brief Values for `prod`, which needs a different input entirely.
                 *
                 * The additive generator is mostly sub-unit, so its running product
                 * flushes to zero long before 256K terms - and zero is bit-stable for
                 * trivial reasons, which would have made the product's determinism case
                 * assert nothing. These sit just either side of 1.0 with a zero-mean
                 * logarithm over their period, so the product stays finite and non-zero
                 * at every size here and the ordering is genuinely observable in it.
                 */
                static float multiplicative_value(std::size_t i)
                {
                    return 1.0f + (static_cast<float>(i % 7) - 3.0f) * 1e-4f;
                }
        };

        // Guards this file's own premise. If a retune ever moved the CPU threshold past
        // kLarge, every case below would silently test the sequential path twice and the
        // parallel one never - and still pass.
        TEST_F(Unit_ReductionDeterminism, BothShapesAreReachedHere)
        {
            auto& q = TestEnvironment::get_context().get_queue();
            const Device::Profile& p = Device::profile_for(q);

            EXPECT_TRUE(is_two_stage(kLarge))
                << "kLarge must exercise the two-stage shape.";

            if (p.is_cpu)
            {
                EXPECT_FALSE(is_two_stage(kSmall))
                    << "kSmall must exercise the single-pass shape on a CPU.";
            }
        }

        TEST_F(Unit_ReductionDeterminism, SumIsBitwiseStable)
        {
            auto op = [](Engine& e, const Tensor& t, Tensor& r) { e.reductions().sum(t, r); };
            expect_bitwise_stable(kSmall, op, additive_value);
            expect_bitwise_stable(kLarge, op, additive_value);
        }

        TEST_F(Unit_ReductionDeterminism, MeanIsBitwiseStable)
        {
            auto op = [](Engine& e, const Tensor& t, Tensor& r) { e.reductions().mean(t, r); };
            expect_bitwise_stable(kSmall, op, additive_value);
            expect_bitwise_stable(kLarge, op, additive_value);
        }

        TEST_F(Unit_ReductionDeterminism, AsumIsBitwiseStable)
        {
            auto op = [](Engine& e, const Tensor& t, Tensor& r) { e.reductions().asum(t, r); };
            expect_bitwise_stable(kSmall, op, additive_value);
            expect_bitwise_stable(kLarge, op, additive_value);
        }

        TEST_F(Unit_ReductionDeterminism, Nrm2IsBitwiseStable)
        {
            auto op = [](Engine& e, const Tensor& t, Tensor& r) { e.reductions().nrm2(t, r); };
            expect_bitwise_stable(kSmall, op, additive_value);
            expect_bitwise_stable(kLarge, op, additive_value);
        }

        TEST_F(Unit_ReductionDeterminism, ProdIsBitwiseStable)
        {
            auto op = [](Engine& e, const Tensor& t, Tensor& r) { e.reductions().prod(t, r); };
            expect_bitwise_stable(kSmall, op, multiplicative_value);
            expect_bitwise_stable(kLarge, op, multiplicative_value);
        }

        /**
        * @brief The accuracy Neumaier compensation buys, on an input built to lose it.
        *
        * One large magnitude followed by many small ones. In `float`, 1e8 has an ulp of
        * 8, so `1e8 + 1` rounds straight back to 1e8 and a naive running sum discards
        * every single one of the 4096 ones that follow - it returns exactly 1e8, off by
        * the entire tail. The compensation term holds what each addition dropped and
        * gives back 100004096, which is itself exactly representable (it is a multiple
        * of 8), so this can assert an exact value rather than a tolerance.
        *
        * The gap between right and wrong here is 4096, not an ulp, which is what makes
        * this a test that cannot pass by accident. Verified to fail without the
        * compensation by removing it: the result came back as 1e8.
        */
        TEST_F(Unit_ReductionDeterminism, CompensatedSumRecoversASmallTailUnderALargeHead)
        {
            constexpr int64_t kOnes = 4096;
            constexpr float kHead = 1e8f;

            auto t = engine->create_tensor({kOnes + 1});
            std::vector<float> host(static_cast<std::size_t>(kOnes + 1), 1.0f);
            host[0] = kHead;
            TestEnvironment::get_context().get_queue()
                .memcpy(t.data(), host.data(), host.size() * sizeof(float)).wait();

            auto res = engine->create_tensor({1});
            engine->reductions().sum(t, res);
            engine->execute().wait();

            EXPECT_FLOAT_EQ(kHead + static_cast<float>(kOnes), scalar(res))
                << "Naive summation returns " << kHead << " here, because every 1.0 falls "
                   "below the ulp of the running total. Neumaier compensation is what "
                   "recovers the tail.";
        }

        /**
        * @brief The same property on the classic input, and against the naive answer.
        *
        * A long run of 0.1f - a value with no exact binary representation, so every
        * addition rounds. Rather than pinning a magic constant, this computes both the
        * exact answer (in double, over the *float* values actually stored) and what a
        * naive float pass would produce, and asserts the device beat the naive one.
        *
        * That comparison is the part with teeth. On the single-pass shape the device
        * kernel adds the same values in the same order as `naive` below, so without the
        * compensation the two are bit-identical and `EXPECT_LT` fails outright.
        */
        TEST_F(Unit_ReductionDeterminism, CompensatedSumBeatsNaiveSummationOnRepeatedTenths)
        {
            for (int64_t n : {kSmall, kLarge})
            {
                SCOPED_TRACE(is_two_stage(n) ? "two-stage shape" : "single-pass shape");

                auto t = filled(n, 0.1f);
                auto res = engine->create_tensor({1});
                engine->reductions().sum(t, res);
                engine->execute().wait();

                // The exact sum of the float values that are actually in the buffer:
                // 0.1f is 0.100000001490116... and n copies of it sum to n times that.
                const double exact = static_cast<double>(n) * static_cast<double>(0.1f);

                float naive = 0.0f;
                for (int64_t i = 0; i < n; ++i) naive += 0.1f;

                const double got_err = std::fabs(static_cast<double>(scalar(res)) - exact);
                const double naive_err = std::fabs(static_cast<double>(naive) - exact);

                EXPECT_LT(got_err, naive_err)
                    << "n = " << n << ": the compensated reduction produced "
                    << scalar(res) << " (error " << got_err << ") but naive float "
                    << "summation produces " << naive << " (error " << naive_err
                    << "). Equal errors mean the compensation is not running.";

                // And it is not merely better, it is right: within one ulp of the exact
                // answer at this magnitude.
                EXPECT_LT(got_err, exact * 1e-7)
                    << "n = " << n << ": compensated summation should land within an ulp.";
            }
        }

        /**
        * @brief The two-stage kernel, driven directly, against the sequential one.
        *
        * `Reduce::reduce` picks its shape from the device, so on this machine the public
        * `reductions().sum` path only reaches the two-stage kernel above 16K elements
        * and the sequential one below. This calls both at the *same* small size - by
        * handing the two-stage path a scratch buffer it would not otherwise get - so the
        * work-group tree is checked against an answer that does not depend on it.
        *
        * Worth having because the tree is the part a CPU under-exercises: on a GPU it is
        * the only path there is, and this project has no GPU to run it on.
        */
        TEST_F(Unit_ReductionDeterminism, TwoStageTreeAgreesWithTheSequentialPass)
        {
            constexpr std::size_t kN = 3000;   // not a multiple of any group width

            auto& ctx = TestEnvironment::get_context();
            auto& q = ctx.get_queue();

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
                for (float v : host) s += static_cast<double>(v);
                return s;
            }();

            // Not bit-equal to each other - they sum in different orders, which is the
            // point of having two shapes - but both must be right, and both must be
            // stable run to run, which the suites above assert separately.
            EXPECT_NEAR(static_cast<double>(scalar(out_tree)), exact, exact * 1e-6);
            EXPECT_NEAR(static_cast<double>(scalar(out_seq)), exact, exact * 1e-6);
        }

    } // namespace Test
} // namespace SushiBLAS
