/**************************************************************************/
/* test_rotmg.cpp                                                         */
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
 * @file test_rotmg.cpp
 * @brief Tests rotmg on its four outcomes, on both rescalings and across both kernels.
 * @author Mustafa Garip
 */

#include <cmath>
#include <limits>
#include <random>
#include <string>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <type_traits>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_readback.hpp>
#include <SushiBLAS/tensor_expectations/scalar_dtype.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
#include "support/tolerance.hpp"
#include "blas_rows.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief Seed of every random draw in this file.
            constexpr std::uint32_t K_SEED = 20261007u;

            /// @brief Elements of a rotmg parameter tensor: the flag and the four entries of H.
            constexpr int64_t K_PARAMETERS = 5;

            /// @brief Preset of every parameter slot; a slot the flag implies must keep it.
            constexpr double K_UNWRITTEN = 99.0;

            /// @brief The factor xROTMG rescales by.
            constexpr double K_GAMMA = 4096.0;

            /// @brief The square of K_GAMMA, the top of the xROTMG scale window.
            constexpr double K_GAMMA_SQUARED = 16777216.0;

            /// @brief Binary exponent of a scale factor above the window.
            constexpr int K_LARGE_EXPONENT = 26;

            /// @brief Binary exponent of a scale factor below the window.
            constexpr int K_SMALL_EXPONENT = -30;

            /// @brief Random quadruples the two kernels are compared on, per element type.
            constexpr int K_RANDOM_QUADRUPLES = 100;

            /// @brief Smallest value of a drawn operand; every draw is positive.
            constexpr double K_DRAW_MINIMUM = 0.125;

            /// @brief Largest value of a drawn operand.
            constexpr double K_DRAW_MAXIMUM = 8.0;

            /**
             * @brief Roundoffs of the larger value the two kernels may differ by: two units in
             *        the last place, each at most two roundoffs.
             */
            constexpr std::size_t K_MODE_ROUNDOFFS = 4;

            /** @brief Holds the four scalars one call of rotmg receives. */
            struct Operands
            {
                double d1; /**< The first scale factor. */
                double d2; /**< The second scale factor. */
                double x1; /**< The first component. */
                double y1; /**< The second component. */
            };

            /** @brief Holds one call of rotmg and what it must leave. */
            struct OutcomeRow
            {
                Operands in;                /**< What the call receives. */
                double param[K_PARAMETERS]; /**< The parameter tensor after the call. */
                double d1;                  /**< The first scale factor after the call. */
                double d2;                  /**< The second scale factor after the call. */
                double x1;                  /**< The first component after the call. */
            };

            /// @brief The four outcomes of xROTMG, each exact in FLOAT32 and FLOAT64.
            constexpr OutcomeRow K_OUTCOMES[] = {
                {{1.0, 1.0, 1.0, 1.0}, {1.0, 1.0, K_UNWRITTEN, K_UNWRITTEN, 1.0}, 0.5, 0.5, 2.0},
                {{5.0, 5.0, 4.0, 2.0}, {0.0, K_UNWRITTEN, -0.5, 0.5, K_UNWRITTEN}, 4.0, 4.0, 5.0},
                {{3.0, 0.0, 5.0, 7.0},
                 {-2.0, K_UNWRITTEN, K_UNWRITTEN, K_UNWRITTEN, K_UNWRITTEN}, 3.0, 0.0, 5.0},
                {{-1.0, 1.0, 1.0, 1.0}, {-1.0, 0.0, 0.0, 0.0, 0.0}, 0.0, 0.0, 0.0}};

            /// @brief The suffixes of the rows of K_OUTCOMES, in order: what each outcome is.
            constexpr const char* K_OUTCOME_WORDS[] = {"FlagMinusOne", "FlagZero", "FlagMinusTwo",
                                                       "NegativeD1"};

            /** @brief Holds every operand of one call as read after the graph ran. */
            template<typename T>
            struct Outcome
            {
                T d1;                 /**< The first scale factor. */
                T d2;                 /**< The second scale factor. */
                T x1;                 /**< The first component. */
                T y1;                 /**< The second component, which a call only reads. */
                std::vector<T> param; /**< The parameter tensor. */
            };

            /** @brief Returns the entries of an array of doubles in the element type T. */
            template<typename T, std::size_t N>
            [[nodiscard]] std::vector<T> as_elements(const double (&values)[N])
            {
                std::vector<T> out;
                out.reserve(N);
                for (const double value : values)
                    out.push_back(static_cast<T>(value));
                return out;
            }

            /**
             * @brief Expects actual within K_MODE_ROUNDOFFS roundoffs of expected, the roundoff
             *        taken at the larger magnitude of the two.
             */
            template<typename R>
            void expect_within_roundoffs(R actual, R expected, const char* what)
            {
                const double a = static_cast<double>(actual);
                const double e = static_cast<double>(expected);
                EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_error(
                    a, e, static_cast<double>(K_MODE_ROUNDOFFS) * TestSupport::unit_roundoff<R>() *
                              std::max(std::abs(a), std::abs(e)))))
                    << what;
            }

            /** @brief Expects two outcomes of one quadruple to agree within the mode bound. */
            template<typename T>
            void expect_outcomes_agree(const Outcome<T>& device, const Outcome<T>& reference)
            {
                expect_within_roundoffs<T>(device.d1, reference.d1, "d1");
                expect_within_roundoffs<T>(device.d2, reference.d2, "d2");
                expect_within_roundoffs<T>(device.x1, reference.x1, "x1");

                ASSERT_EQ(device.param.size(), reference.param.size());
                for (std::size_t slot = 0; slot < device.param.size(); ++slot)
                    expect_within_roundoffs<T>(device.param[slot], reference.param[slot], "param");
            }
        } // namespace

        /** @brief Holds the index of the row of K_OUTCOMES that completes a row of the suite. */
        struct OutcomeAxis
        {
            std::size_t index; /**< The row of K_OUTCOMES the case checks. */
        };

        /** @brief Returns the tail of a row named by the outcome it checks. */
        [[nodiscard]] inline std::string axis_name(const OutcomeAxis& axis)
        {
            return std::string("_") + K_OUTCOME_WORDS[axis.index];
        }

        /**
         * @brief Records rotmg on scalar tensors and reads them back.
         */
        class RotmgStart : public BlasEngineStart
        {
            protected:
                /** @brief Returns a rank-1 tensor of count elements that all hold value. */
                template<typename T>
                [[nodiscard]] Tensor filled(int64_t count, double value)
                {
                    Tensor tensor = engine->create_tensor({count}, sushiblas_dtype<T>());
                    const std::vector<T> values(static_cast<std::size_t>(count),
                                                static_cast<T>(value));
                    TestEnvironment::queue()
                        .memcpy(tensor.data(), values.data(), values.size() * sizeof(T)).wait();
                    return tensor;
                }

                /**
                 * @brief Returns what each call leaves, all recorded in one graph with param
                 *        preset to K_UNWRITTEN.
                 */
                template<typename T>
                [[nodiscard]] std::vector<Outcome<T>> transform(const std::vector<Operands>& calls)
                {
                    std::vector<Tensor> d1;
                    std::vector<Tensor> d2;
                    std::vector<Tensor> x1;
                    std::vector<Tensor> y1;
                    std::vector<Tensor> param;
                    d1.reserve(calls.size());
                    d2.reserve(calls.size());
                    x1.reserve(calls.size());
                    y1.reserve(calls.size());
                    param.reserve(calls.size());

                    SushiBLAS::BLASOps ops(recorder());
                    for (const Operands& call : calls)
                    {
                        d1.push_back(filled<T>(1, call.d1));
                        d2.push_back(filled<T>(1, call.d2));
                        x1.push_back(filled<T>(1, call.x1));
                        y1.push_back(filled<T>(1, call.y1));
                        param.push_back(filled<T>(K_PARAMETERS, K_UNWRITTEN));
                        ops.rotmg(d1.back(), d2.back(), x1.back(), y1.back(), param.back());
                    }
                    recording->run().value();

                    std::vector<Outcome<T>> outcomes;
                    outcomes.reserve(calls.size());
                    for (std::size_t i = 0; i < calls.size(); ++i)
                        outcomes.push_back(Outcome<T>{read_tensor<T>(d1[i])[0],
                                                      read_tensor<T>(d2[i])[0],
                                                      read_tensor<T>(x1[i])[0],
                                                      read_tensor<T>(y1[i])[0],
                                                      read_tensor<T>(param[i])});
                    return outcomes;
                }

                /**
                 * @brief Expects the rescaled H of (factor, factor, 4, 2) in the kernel of the row.
                 * @param gamma What the entries of H and x1 are multiplied by.
                 * @param scaled What both scale factors become.
                 */
                template<typename T>
                void expect_rescaled(T factor, T gamma, T scaled)
                {
                    const std::vector<T> expected = {T(-1), gamma, T(-0.5) * gamma,
                                                     T(0.5) * gamma, gamma};

                    const Outcome<T> outcome = transform<T>(
                        {Operands{static_cast<double>(factor), static_cast<double>(factor),
                                  4.0, 2.0}}).front();

                    EXPECT_EQ(outcome.param, expected);
                    EXPECT_EQ(outcome.d1, scaled);
                    EXPECT_EQ(outcome.d2, scaled);
                    EXPECT_EQ(outcome.x1, T(5) * gamma);
                    EXPECT_EQ(outcome.y1, T(2));
                }

                /** @brief Returns the operands of the random quadruples both kernels run. */
                [[nodiscard]] static std::vector<Operands> random_quadruples()
                {
                    std::mt19937 generator(K_SEED);
                    std::uniform_real_distribution<double> distribution(K_DRAW_MINIMUM,
                                                                        K_DRAW_MAXIMUM);
                    std::vector<Operands> calls(static_cast<std::size_t>(K_RANDOM_QUADRUPLES));
                    for (Operands& call : calls)
                    {
                        call.d1 = distribution(generator);
                        call.d2 = distribution(generator);
                        call.x1 = distribution(generator);
                        call.y1 = distribution(generator);
                    }
                    return calls;
                }

                /** @brief Expects the device and the reference outcomes of all calls to agree. */
                template<typename T>
                static void expect_all_agree(const std::vector<Outcome<T>>& device,
                                             const std::vector<Outcome<T>>& reference)
                {
                    ASSERT_EQ(device.size(), reference.size());
                    for (std::size_t i = 0; i < device.size(); ++i)
                    {
                        SCOPED_TRACE(::testing::Message() << "quadruple " << i);
                        expect_outcomes_agree(device[i], reference[i]);
                    }
                }
        };

        /** @brief Runs one outcome of xROTMG per row, over the element types, layouts and kernels. */
        class Medium_RotmgOutcomes : public BlasRowCase<RotmgStart, OutcomeAxis>
        {

        };

        /** @brief Runs the rescalings of xROTMG per row, over the element types, layouts and kernels. */
        class Medium_Rotmg : public BlasRowCase<RotmgStart, NoAxis>
        {

        };

        /** @brief Runs the comparison of the two kernels per layout and element type. */
        class Medium_RotmgKernels : public BlasRowCase<RotmgStart, NoAxis>
        {

        };

        /** @brief Returns the element types rotmg accepts. */
        [[nodiscard]] inline std::vector<Core::DataType> rotmg_types()
        {
            return {Core::DataType::Float32, Core::DataType::Float64};
        }

        /**
         * @brief Tests one outcome of xROTMG and which parameters it writes.
         */
        TEST_P(Medium_RotmgOutcomes, EachOutcomeStoresOnlyWhatItsFlagDoesNotImply)
        {
            const OutcomeRow& row = K_OUTCOMES[GetParam().axis.index];
            with_real_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;

                const Outcome<T> outcome = transform<T>({row.in}).front();

                EXPECT_EQ(outcome.param, as_elements<T>(row.param));
                EXPECT_EQ(outcome.d1, static_cast<T>(row.d1));
                EXPECT_EQ(outcome.d2, static_cast<T>(row.d2));
                EXPECT_EQ(outcome.x1, static_cast<T>(row.x1));
                EXPECT_EQ(outcome.y1, static_cast<T>(row.in.y1));
            });
        }

        INSTANTIATE_TEST_SUITE_P(
            , Medium_RotmgOutcomes,
            ::testing::ValuesIn(rows_of(rotmg_types(),
                                        {ExecutionMode::ForceReference, ExecutionMode::ForceSYCL},
                                        std::vector<OutcomeAxis>{{0}, {1}, {2}, {3}})),
            row_name<OutcomeAxis>);

        /**
         * @brief Tests that scale factors above the window are divided back into it.
         */
        TEST_P(Medium_Rotmg, LargeFactorsAreRescaledDown)
        {
            with_real_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;

                const T factor = std::ldexp(T(1), K_LARGE_EXPONENT);
                const T scaled = (factor / T(1.25)) / static_cast<T>(K_GAMMA_SQUARED);

                expect_rescaled<T>(factor, static_cast<T>(K_GAMMA), scaled);
            });
        }

        /**
         * @brief Tests that scale factors below the window are multiplied back into it.
         */
        TEST_P(Medium_Rotmg, SmallFactorsAreRescaledUp)
        {
            with_real_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;

                const T factor = std::ldexp(T(1), K_SMALL_EXPONENT);
                const T scaled = (factor / T(1.25)) * static_cast<T>(K_GAMMA_SQUARED);

                expect_rescaled<T>(factor, T(1) / static_cast<T>(K_GAMMA), scaled);
            });
        }

        INSTANTIATE_TEST_SUITE_P(
            , Medium_Rotmg,
            ::testing::ValuesIn(plain_rows(rotmg_types(), {ExecutionMode::ForceReference,
                                                           ExecutionMode::ForceSYCL})),
            row_name<NoAxis>);

        /**
         * @brief Tests that the two kernels give transformations within K_MODE_ROUNDOFFS
         *        roundoffs of each other on positive operands.
         *
         * The row names the layout and the element type; its kernel is the device one, and the
         * reference run is made inside the case.
         */
        TEST_P(Medium_RotmgKernels, BothModesAgree)
        {
            with_real_element(GetParam().type, [&](auto element)
            {
                using T = typename decltype(element)::type;

                const std::vector<Operands> calls = random_quadruples();

                start(GetParam().layout, ExecutionMode::ForceReference);
                const auto reference = transform<T>(calls);
                start(GetParam().layout, ExecutionMode::ForceSYCL);
                const auto device = transform<T>(calls);

                expect_all_agree(device, reference);
            });
        }

        INSTANTIATE_TEST_SUITE_P(
            , Medium_RotmgKernels,
            ::testing::ValuesIn(plain_rows(rotmg_types(), {ExecutionMode::ForceSYCL})),
            row_name<NoAxis>);
    } // namespace Test
} // namespace SushiBLAS
