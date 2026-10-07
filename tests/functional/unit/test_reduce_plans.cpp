/**************************************************************************/
/* test_reduce_plans.cpp                                                  */
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
 * @file test_reduce_plans.cpp
 * @brief Tests the plan-taking forms of Reduce::reduce and Reduce::reduce_indexed.
 * @author Mustafa Garip
 */

#include <limits>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <SushiBLAS/SushiBLAS.h>
#include <SushiBLAS/support/reduce_1d.hpp>
#include <SushiBLAS/support/device_policy.hpp>
#include "../../common/test_utils.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief Holds the element count of the injected-plan cases.
            constexpr std::size_t K_ELEMENTS = 1000;

            /// @brief Holds an element count below the work-items of a three-by-eight plan.
            constexpr std::size_t K_FEW_ELEMENTS = 17;

            /// @brief Holds an element count past the CPU's two-stage threshold.
            constexpr std::size_t K_TWO_STAGE_ELEMENTS = 100000;

            /// @brief Holds a value no case of this file reduces to.
            constexpr int64_t K_UNWRITTEN = -1;

            /// @brief Holds the (value, index) partial the index case folds.
            using Candidate = Reduce::IndexedValue<float>;
        }

        /// @brief Runs the reductions whose launch geometry the test chooses.
        class Unit_ReducePlans : public SushiBLASTest
        {
            protected:
                /** @brief Returns the queue every case submits to. */
                static sycl::queue& queue()
                {
                    return TestEnvironment::get_context().get_queue();
                }

                /** @brief Returns the exact INT64 sum of the first n values under geometry. */
                int64_t planned_sum(const Tensor& in, std::size_t n, const Reduce::Plan& geometry)
                {
                    auto out = engine->create_tensor({1}, Core::DataType::Int64);
                    auto partials = engine->create_tensor({8}, Core::DataType::Int64);
                    fill_tensor(out, {static_cast<float>(K_UNWRITTEN)});

                    Reduce::reduce<Reduce::ExactSum>(
                        queue(), geometry, in.data_as<int64_t>(), out.data_as<int64_t>(), n,
                        partials.data_as<int64_t>(),
                        [](int64_t value) { return value; },
                        [](int64_t value, [[maybe_unused]] std::size_t count) { return value; },
                        {}).wait();

                    return read_tensor<int64_t>(out)[0];
                }

                /** @brief Returns the index RunningArgMax elects over values under geometry. */
                int64_t planned_argmax(const Tensor& in, std::size_t n,
                                       const Reduce::Plan& geometry)
                {
                    auto out = engine->create_tensor({1}, Core::DataType::Int64);
                    auto partials = engine->create_tensor({8}, Core::DataType::Int64);
                    fill_tensor(out, {static_cast<float>(K_UNWRITTEN)});

                    // Eight INT64 elements are the 64 bytes four candidates take.
                    static_assert(4 * sizeof(Candidate) <= 8 * sizeof(int64_t),
                                  "The partials tensor must hold one candidate per work-group.");

                    const float* values = in.data_as<float>();
                    Reduce::reduce_indexed<Reduce::RunningArgMax>(
                        queue(), geometry, n,
                        [values](std::size_t i)
                        {
                            return Candidate{values[i], static_cast<std::int64_t>(i)};
                        },
                        out.data_as<int64_t>(), static_cast<Candidate*>(partials.data()),
                        [](Candidate best, [[maybe_unused]] std::size_t count)
                        {
                            return best.index;
                        },
                        {}).wait();

                    return read_tensor<int64_t>(out)[0];
                }
        };

        /**
         * @brief Tests that every element is folded once under each injected geometry.
         */
        TEST_F(Unit_ReducePlans, AnInjectedTwoStagePlanFoldsEveryElement)
        {
            std::vector<int64_t> values(K_ELEMENTS);
            int64_t expected = 0;
            int64_t expected_few = 0;
            for (std::size_t i = 0; i < K_ELEMENTS; ++i)
            {
                values[i] = static_cast<int64_t>(i) * 3 - 500;
                expected += values[i];
                if (i < K_FEW_ELEMENTS)
                    expected_few += values[i];
            }

            auto in = create_initialized_tensor<int64_t>(
                {static_cast<int64_t>(K_ELEMENTS)}, values);

            EXPECT_EQ(expected, planned_sum(in, K_ELEMENTS, Reduce::Plan{3, 8}));
            EXPECT_EQ(expected, planned_sum(in, K_ELEMENTS, Reduce::Plan{2, 4}));
            EXPECT_EQ(expected, planned_sum(in, K_ELEMENTS, Reduce::Plan{0, 1}));

            // Seventeen elements leave seven of the twenty-four work-items with nothing to fold.
            EXPECT_EQ(expected_few, planned_sum(in, K_FEW_ELEMENTS, Reduce::Plan{3, 8}));
        }

        /**
         * @brief Tests that a two-stage geometry and the single pass elect the same index.
         */
        TEST_F(Unit_ReducePlans, AnInjectedPlanElectsTheSameIndex)
        {
            constexpr std::size_t first_peak = 123;
            constexpr std::size_t second_peak = 777;
            constexpr std::size_t nan_at = 500;

            // The maximum appears twice and a NaN sits between its two positions.
            std::vector<float> values(K_ELEMENTS, 1.0f);
            values[first_peak] = 9.0f;
            values[second_peak] = 9.0f;
            values[nan_at] = std::numeric_limits<float>::quiet_NaN();

            auto in = create_initialized_tensor<float>({static_cast<int64_t>(K_ELEMENTS)}, values);

            const int64_t staged = planned_argmax(in, K_ELEMENTS, Reduce::Plan{4, 8});
            const int64_t single = planned_argmax(in, K_ELEMENTS, Reduce::Plan{0, 1});

            EXPECT_EQ(static_cast<int64_t>(first_peak), staged);
            EXPECT_EQ(staged, single);
        }

        /**
         * @brief Tests that the queue form and the plan form give the same bits under one plan.
         */
        TEST_F(Unit_ReducePlans, TheQueueFormMatchesThePlanItComputes)
        {
            std::vector<float> values(K_TWO_STAGE_ELEMENTS);
            for (std::size_t i = 0; i < values.size(); ++i)
                values[i] = 0.1f + static_cast<float>(i % 97) * 0.017f;

            auto in = create_initialized_tensor<float>(
                {static_cast<int64_t>(K_TWO_STAGE_ELEMENTS)}, values);
            auto partials = engine->create_tensor({static_cast<int64_t>(Reduce::K_MAX_GROUPS)});
            auto by_queue = engine->create_tensor({1});
            auto by_plan = engine->create_tensor({1});
            fill_tensor(by_queue, {0.0f});
            fill_tensor(by_plan, {0.0f});

            const auto identity = [](float value) { return value; };
            const auto keep = [](float value, [[maybe_unused]] std::size_t count) { return value; };

            Reduce::reduce<Reduce::CompensatedSum>(
                queue(), in.data_as<float>(), by_queue.data_as<float>(), K_TWO_STAGE_ELEMENTS,
                partials.data_as<float>(), Reduce::K_MAX_GROUPS, identity, keep, {}).wait();

            const Reduce::Plan geometry = Reduce::plan(
                Device::profile_for(queue()), K_TWO_STAGE_ELEMENTS, Reduce::K_MAX_GROUPS);

            Reduce::reduce<Reduce::CompensatedSum>(
                queue(), geometry, in.data_as<float>(), by_plan.data_as<float>(),
                K_TWO_STAGE_ELEMENTS, partials.data_as<float>(), identity, keep, {}).wait();

            const float queue_sum = read_tensor<float>(by_queue)[0];
            const float plan_sum = read_tensor<float>(by_plan)[0];

            ASSERT_NE(0.0f, queue_sum) << "the queue form wrote nothing";
            EXPECT_EQ(0, std::memcmp(&queue_sum, &plan_sum, sizeof(float)))
                << "the queue form gave " << queue_sum << ", the plan form " << plan_sum;
        }
    } // namespace Test
} // namespace SushiBLAS
