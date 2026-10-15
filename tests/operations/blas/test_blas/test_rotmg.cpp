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
#include <SushiBLAS/typed_suite_naming/default_type_names.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
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

            /// @brief Units in the last place the two kernels may differ by.
            constexpr double K_MODE_ULPS = 2.0;

            /// @brief Both kernels an operation has.
            constexpr ExecutionMode K_MODES[] = {ExecutionMode::ForceReference,
                                                 ExecutionMode::ForceSYCL};

            /// @brief Both element orders an engine can create tensors in.
            constexpr Core::Layout K_LAYOUTS[] = {Core::Layout::RowMajor,
                                                  Core::Layout::ColumnMajor};

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

            /** @brief Returns the name of a mode and a layout for a failure message. */
            [[nodiscard]] std::string engine_name(Core::Layout layout, ExecutionMode mode)
            {
                return std::string(mode == ExecutionMode::ForceSYCL ? "ForceSYCL"
                                                                     : "ForceReference") +
                       (layout == Core::Layout::RowMajor ? ", ROW_MAJOR" : ", COLUMN_MAJOR");
            }

            /** @brief Returns whether the test device computes in T. */
            template<typename T>
            [[nodiscard]] bool device_runs()
            {
                if constexpr (std::is_same_v<T, double>)
                    return TestEnvironment::queue().get_device().has(
                        sycl::aspect::fp64);
                else
                    return true;
            }

            /** @brief Returns the spacing of the values of R at a magnitude. */
            template<typename R>
            [[nodiscard]] double ulp_at(double magnitude)
            {
                const int exponent = std::max(std::ilogb(magnitude),
                                              std::numeric_limits<R>::min_exponent - 1);
                return std::ldexp(1.0, exponent - (std::numeric_limits<R>::digits - 1));
            }

            /**
             * @brief Expects actual within K_MODE_ULPS units in the last place of expected, the
             *        unit taken at the larger magnitude of the two.
             */
            template<typename R>
            void expect_within_ulps(R actual, R expected, const char* what)
            {
                const double a = static_cast<double>(actual);
                const double e = static_cast<double>(expected);
                const double unit = ulp_at<R>(std::max(std::abs(a), std::abs(e)));
                EXPECT_LE(std::abs(a - e), K_MODE_ULPS * unit) << what;
            }
        } // namespace

        /**
         * @brief Records rotmg on scalar tensors of element type T and reads them back.
         */
        template<typename T>
        class Medium_Rotmg : public SushiBLASTest
        {
            protected:
                /** @brief Holds every operand of one call as read after the graph ran. */
                struct Outcome
                {
                    T d1;                 /**< The first scale factor. */
                    T d2;                 /**< The second scale factor. */
                    T x1;                 /**< The first component. */
                    T y1;                 /**< The second component, which a call only reads. */
                    std::vector<T> param; /**< The parameter tensor. */
                };

                /** @brief Skips the case on a device that does not compute in T. */
                void SetUp() override
                {
                    SushiBLASTest::SetUp();
                    if (!device_runs<T>())
                        GTEST_SKIP() << "the device does not compute in this element type";
                }

                /** @brief Rebuilds the engine in a layout and selects a kernel. */
                void start(Core::Layout layout, ExecutionMode mode)
                {
                    this->reinit_engine(layout);
                    this->engine->set_mode(mode);
                }

                /** @brief Returns a rank-1 tensor of count elements that all hold value. */
                [[nodiscard]] Tensor filled(int64_t count, double value)
                {
                    Tensor tensor = this->engine->create_tensor({count}, sushiblas_dtype<T>());
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
                [[nodiscard]] std::vector<Outcome> transform(const std::vector<Operands>& calls)
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

                    SushiBLAS::BLASOps ops(*this->engine);
                    for (const Operands& call : calls)
                    {
                        d1.push_back(filled(1, call.d1));
                        d2.push_back(filled(1, call.d2));
                        x1.push_back(filled(1, call.x1));
                        y1.push_back(filled(1, call.y1));
                        param.push_back(filled(K_PARAMETERS, K_UNWRITTEN));
                        ops.rotmg(d1.back(), d2.back(), x1.back(), y1.back(), param.back());
                    }
                    this->engine->execute().value();

                    std::vector<Outcome> outcomes;
                    outcomes.reserve(calls.size());
                    for (std::size_t i = 0; i < calls.size(); ++i)
                        outcomes.push_back(Outcome{read_tensor<T>(d1[i])[0],
                                                   read_tensor<T>(d2[i])[0],
                                                   read_tensor<T>(x1[i])[0],
                                                   read_tensor<T>(y1[i])[0],
                                                   read_tensor<T>(param[i])});
                    return outcomes;
                }

                /**
                 * @brief Expects the rescaled H of (factor, factor, 4, 2) in both kernels.
                 * @param gamma What the entries of H and x1 are multiplied by.
                 * @param scaled What both scale factors become.
                 */
                void expect_rescaled(T factor, T gamma, T scaled)
                {
                    const std::vector<T> expected = {T(-1), gamma, T(-0.5) * gamma,
                                                     T(0.5) * gamma, gamma};

                    for (const Core::Layout layout : K_LAYOUTS)
                    {
                        for (const ExecutionMode mode : K_MODES)
                        {
                            SCOPED_TRACE(engine_name(layout, mode));
                            start(layout, mode);
                            const Outcome outcome = transform(
                                {Operands{static_cast<double>(factor),
                                          static_cast<double>(factor), 4.0, 2.0}}).front();

                            ASSERT_EQ(outcome.param.size(), expected.size());
                            for (std::size_t slot = 0; slot < expected.size(); ++slot)
                                EXPECT_EQ(outcome.param[slot], expected[slot])
                                    << "param[" << slot << "]";
                            EXPECT_EQ(outcome.d1, scaled);
                            EXPECT_EQ(outcome.d2, scaled);
                            EXPECT_EQ(outcome.x1, T(5) * gamma);
                            EXPECT_EQ(outcome.y1, T(2));
                        }
                    }
                }
        };

        /// @brief Names the element types rotmg accepts.
        using RotmgTypes = ::testing::Types<float, double>;
        SB_TYPED_TEST_SUITE(Medium_Rotmg, RotmgTypes);

        /**
         * @brief Tests the four outcomes of xROTMG and which parameters each one writes.
         */
        TYPED_TEST(Medium_Rotmg, EachOutcomeStoresOnlyWhatItsFlagDoesNotImply)
        {
            using T = TypeParam;

            std::vector<Operands> calls;
            for (const OutcomeRow& row : K_OUTCOMES)
                calls.push_back(row.in);

            for (const Core::Layout layout : K_LAYOUTS)
            {
                for (const ExecutionMode mode : K_MODES)
                {
                    SCOPED_TRACE(engine_name(layout, mode));
                    this->start(layout, mode);
                    const auto outcomes = this->transform(calls);

                    for (std::size_t i = 0; i < outcomes.size(); ++i)
                    {
                        const OutcomeRow& row = K_OUTCOMES[i];
                        SCOPED_TRACE(::testing::Message()
                                     << "d1 " << row.in.d1 << ", d2 " << row.in.d2 << ", x1 "
                                     << row.in.x1 << ", y1 " << row.in.y1);

                        ASSERT_EQ(outcomes[i].param.size(),
                                  static_cast<std::size_t>(K_PARAMETERS));
                        for (std::size_t slot = 0; slot < outcomes[i].param.size(); ++slot)
                            EXPECT_EQ(outcomes[i].param[slot], static_cast<T>(row.param[slot]))
                                << "param[" << slot << "]";

                        EXPECT_EQ(outcomes[i].d1, static_cast<T>(row.d1));
                        EXPECT_EQ(outcomes[i].d2, static_cast<T>(row.d2));
                        EXPECT_EQ(outcomes[i].x1, static_cast<T>(row.x1));
                        EXPECT_EQ(outcomes[i].y1, static_cast<T>(row.in.y1));
                    }
                }
            }
        }

        /**
         * @brief Tests that scale factors above the window are divided back into it.
         */
        TYPED_TEST(Medium_Rotmg, LargeFactorsAreRescaledDown)
        {
            using T = TypeParam;
            const T factor = std::ldexp(T(1), K_LARGE_EXPONENT);
            const T scaled = (factor / T(1.25)) / static_cast<T>(K_GAMMA_SQUARED);

            this->expect_rescaled(factor, static_cast<T>(K_GAMMA), scaled);
        }

        /**
         * @brief Tests that scale factors below the window are multiplied back into it.
         */
        TYPED_TEST(Medium_Rotmg, SmallFactorsAreRescaledUp)
        {
            using T = TypeParam;
            const T factor = std::ldexp(T(1), K_SMALL_EXPONENT);
            const T scaled = (factor / T(1.25)) * static_cast<T>(K_GAMMA_SQUARED);

            this->expect_rescaled(factor, T(1) / static_cast<T>(K_GAMMA), scaled);
        }

        /**
         * @brief Tests that the two kernels give transformations within K_MODE_ULPS of each
         *        other on positive operands.
         */
        TYPED_TEST(Medium_Rotmg, BothModesAgree)
        {
            using T = TypeParam;

            std::mt19937 generator(K_SEED);
            std::uniform_real_distribution<double> distribution(K_DRAW_MINIMUM, K_DRAW_MAXIMUM);
            std::vector<Operands> calls(static_cast<std::size_t>(K_RANDOM_QUADRUPLES));
            for (Operands& call : calls)
            {
                call.d1 = distribution(generator);
                call.d2 = distribution(generator);
                call.x1 = distribution(generator);
                call.y1 = distribution(generator);
            }

            for (const Core::Layout layout : K_LAYOUTS)
            {
                SCOPED_TRACE(layout == Core::Layout::RowMajor ? "ROW_MAJOR" : "COLUMN_MAJOR");
                this->start(layout, ExecutionMode::ForceReference);
                const auto reference = this->transform(calls);
                this->start(layout, ExecutionMode::ForceSYCL);
                const auto device = this->transform(calls);

                ASSERT_EQ(device.size(), reference.size());
                for (std::size_t i = 0; i < device.size(); ++i)
                {
                    SCOPED_TRACE(::testing::Message() << "quadruple " << i);
                    expect_within_ulps<T>(device[i].d1, reference[i].d1, "d1");
                    expect_within_ulps<T>(device[i].d2, reference[i].d2, "d2");
                    expect_within_ulps<T>(device[i].x1, reference[i].x1, "x1");

                    ASSERT_EQ(device[i].param.size(), reference[i].param.size());
                    for (std::size_t slot = 0; slot < device[i].param.size(); ++slot)
                        expect_within_ulps<T>(device[i].param[slot], reference[i].param[slot],
                                              "param");
                }
            }
        }
    } // namespace Test
} // namespace SushiBLAS
