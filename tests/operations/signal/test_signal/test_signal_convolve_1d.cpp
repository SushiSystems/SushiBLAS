/**************************************************************************/
/* test_signal_convolve_1d.cpp                                           */
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

/**
 * @file test_signal_convolve_1d.cpp
 * @brief Tests SignalOps::convolve_1d in both forced modes against hand values, an independent
 * formula and a rounding bound, and its refusals.
 * @author Mustafa Garip
 */

#include <cmath>
#include <string>
#include <string_view>
#include <vector>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/signal/signal.hpp>
#include <SushiBLAS/arithmetic/accumulator.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/tensor_comparison.hpp>
#include <SushiBLAS/tensor_expectations/scalar_dtype.hpp>
#include <SushiBLAS/typed_suite_naming/default_type_names.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
#include <SushiBLAS/blas_test_oracle/blas_truth.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief The forced modes every case runs in.
            constexpr ExecutionMode K_FORCED_MODES[] = {ExecutionMode::ForceSYCL,
                                                        ExecutionMode::ForceReference};

            /// @brief The value an output holds before its operation runs.
            constexpr double K_SENTINEL = 7.0;

            /// @brief The largest magnitude up to which HALF holds every integer.
            constexpr double K_HALF_EXACT_INTEGER_LIMIT = 2048.0;

            /// @brief Holds one signal shape and tap count the operation is run on.
            struct ConvolutionCase
            {
                    /**
                     * @brief Returns the output's extents: the signal's with the last one shortened
                     *        to `L - K + 1`.
                     */
                    [[nodiscard]] std::vector<int64_t> output_dims() const
                    {
                        std::vector<int64_t> shortened = dims;
                        shortened.back() = dims.back() - taps + 1;
                        return shortened;
                }

                /** @brief Returns how many rows the signal holds. */
                [[nodiscard]] int64_t rows() const noexcept
                {
                    int64_t count = 1;
                    for (std::size_t i = 0; i + 1 < dims.size(); ++i)
                        count *= dims[i];
                    return count;
                }

                const char* id = "";        /**< The case's name in a failure message. */
                std::vector<int64_t> dims;  /**< The signal's extents; the last is L. */
                int64_t taps = 0;           /**< K. */
            };

            /**
             * @brief Returns the cases every dtype and mode runs: one row, batched rows, K = 1 and
             *        K = L.
             */
            std::vector<ConvolutionCase> table_cases()
            {
                return {
                    {"one row", {16}, 5},
                    {"batched rows", {3, 4, 16}, 5},
                    {"one tap", {2, 7}, 1},
                    {"taps as long as the row", {4, 6}, 6},
                    {"one output sample per row", {2, 3, 9}, 8},
                };
            }

            /** @brief Returns the name a failure message gives a case run in mode. */
            std::string trace_of(const ConvolutionCase& c, ExecutionMode mode)
            {
                return std::string(c.id) +
                       (mode == ExecutionMode::ForceReference ? " ForceReference" : " ForceSYCL");
            }

            /** @brief Returns how many elements a tensor of dims holds. */
            std::size_t element_count(const std::vector<int64_t>& dims) noexcept
            {
                int64_t count = 1;
                for (const int64_t extent : dims)
                    count *= extent;
                return static_cast<std::size_t>(count);
            }

            /** @brief Returns `((i * multiplier) % modulus) - shift` for i below count. */
            std::vector<double> residue_pattern(std::size_t count, int64_t multiplier,
                                                int64_t modulus, int64_t shift)
            {
                std::vector<double> values(count);
                for (std::size_t i = 0; i < count; ++i)
                    values[i] = static_cast<double>(
                        (static_cast<int64_t>(i) * multiplier) % modulus - shift);
                return values;
            }

            /** @brief Returns `0.37 * v + 0.011` for every v; no element type holds it exactly. */
            std::vector<double> fractional(std::vector<double> values)
            {
                for (double& v : values)
                    v = 0.37 * v + 0.011;
                return values;
            }

            /** @brief Returns values rounded to T. */
            template<typename T>
            std::vector<T> elements_from(const std::vector<double>& values)
            {
                std::vector<T> elements(values.size());
                for (std::size_t i = 0; i < values.size(); ++i)
                {
                    if constexpr (std::is_same_v<T, sycl::half>)
                        elements[i] = sycl::half(static_cast<float>(values[i]));
                    else
                        elements[i] = static_cast<T>(values[i]);
                }
                return elements;
            }

            /** @brief Returns elements widened to double. */
            template<typename T>
            std::vector<double> widened(const std::vector<T>& elements)
            {
                std::vector<double> values(elements.size());
                for (std::size_t i = 0; i < elements.size(); ++i)
                    values[i] = Detail::to_double(elements[i]);
                return values;
            }

            /**
             * @brief Returns a dense tensor of dims holding values rounded to T.
             * @param values One value per element; empty when dims holds a zero.
             */
            template<typename T>
            Tensor tensor_from(Engine& engine, const std::vector<int64_t>& dims,
                               const std::vector<double>& values)
            {
                Tensor t = engine.create_tensor(
                    SushiRuntime::span<const int64_t>(dims.data(), dims.data() + dims.size()),
                    sushiblas_dtype<T>());

                if (t.num_elements > 0)
                    SushiBLAS::IO().from_vector(t, elements_from<T>(values));
                return t;
            }

            /** @brief Returns a dense tensor of dims with every element at K_SENTINEL. */
            template<typename T>
            Tensor sentinel_tensor(Engine& engine, const std::vector<int64_t>& dims)
            {
                return tensor_from<T>(engine, dims,
                                      std::vector<double>(element_count(dims), K_SENTINEL));
            }

            /**
             * @brief Returns the samples the library writes for signal and taps, shaped as the case
             *        says.
             */
            template<typename T>
            std::vector<T> convolution_of(Engine& engine, const ConvolutionCase& c,
                                          const Tensor& signal, const Tensor& taps)
            {
                Tensor out = sentinel_tensor<T>(engine, c.output_dims());
                RecordingFixture graph(engine);
                SushiBLAS::SignalOps(graph.recorder()).convolve_1d(signal, taps, out);
                graph.run().value();
                return SushiBLAS::IO().to_vector<T>(out);
            }

            /**
             * @brief Returns the sums of the formula for the case, as doubles.
             * @note The loop runs over k of `signal[n + K - 1 - k]`, written apart from the
             *       kernels.
             */
            std::vector<double> convolution_by_formula(const ConvolutionCase& c,
                                                       const std::vector<double>& signal,
                                                       const std::vector<double>& taps)
            {
                const int64_t length = c.dims.back();
                const int64_t out_length = length - c.taps + 1;
                std::vector<double> out(element_count(c.output_dims()), 0.0);

                for (int64_t row = 0; row < c.rows(); ++row)
                    for (int64_t n = 0; n < out_length; ++n)
                        for (int64_t k = 0; k < c.taps; ++k)
                            out[static_cast<std::size_t>(row * out_length + n)] +=
                                signal[static_cast<std::size_t>(row * length + n + c.taps - 1 - k)]
                                * taps[static_cast<std::size_t>(k)];
                return out;
            }

            /**
             * @brief Returns the bound on a sample's distance from its exact sum in one mode.
             * @param taps K, the number of products the sample sums.
             * @param truth The sample's exact sum and the sum of its products' magnitudes.
             * @see tests/operations/signal/README.md
             */
            template<typename T>
            double convolution_bound(ExecutionMode mode, int64_t taps, const TruthValue& truth)
            {
                const double store = unit_roundoff<T>() * std::fabs(truth.value);

                if (mode == ExecutionMode::ForceSYCL)
                    return rounding_bound(static_cast<std::size_t>(2 * taps),
                                          unit_roundoff<Kernels::Accumulator<T>>(),
                                          truth.magnitude) + store;

                const std::size_t roundings =
                    std::is_same_v<T, double> ? 5 : static_cast<std::size_t>(taps) + 2;
                return rounding_bound(roundings, unit_roundoff<double>(), truth.magnitude) + store;
            }

            /** @brief Returns true when the test device can compute in T. */
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
             * @brief Expects record to throw std::invalid_argument naming the operation and
             *        operand.
             * @tparam Record Callable `() -> void` that records one malformed call.
             */
            template<typename Record>
            void expect_refusal(Record&& record, const char* operand)
            {
                try
                {
                    record();
                    ADD_FAILURE() << "signal.convolve_1d accepted a malformed '" << operand << "'";
                }
                catch (const std::invalid_argument& error)
                {
                    const std::string text = error.what();
                    EXPECT_NE(text.find("signal.convolve_1d"), std::string::npos) << text;
                    EXPECT_NE(text.find("'" + std::string(operand) + "'"), std::string::npos)
                        << text;
                }
            }
        } // namespace

        /**
         * @brief Runs the convolution in one element type, skipped where the device lacks it.
         */
        template<typename T>
        class Medium_SignalConvolve1DTyped : public SushiBLASTest
        {
            protected:
                /** @brief Builds the engine and skips the case on a device without T's aspect. */
                void SetUp() override
                {
                    SushiBLASTest::SetUp();

                    if (!device_computes<T>())
                        GTEST_SKIP() << "the device does not report the aspect this dtype needs";
                }
        };

        using Convolve1DTypes = ::testing::Types<sycl::half, float, double>;
        SB_TYPED_TEST_SUITE(Medium_SignalConvolve1DTyped, Convolve1DTypes);

        /**
         * @brief Runs the convolution cases that fix one element type.
         */
        class Medium_SignalConvolve1D : public SushiBLASTest
        {
        };

        /**
         * @brief Tests that a two-tap valid convolution gives hand-worked values, the taps applied
         *        in reverse.
         */
        TYPED_TEST(Medium_SignalConvolve1DTyped, TheTwoTapConvolutionGivesItsHandWorkedValues)
        {
            using T = TypeParam;
            Engine& engine = *this->engine;

            const ConvolutionCase c{"hand", {5}, 2};

            for (const ExecutionMode mode : K_FORCED_MODES)
            {
                SCOPED_TRACE(trace_of(c, mode));
                engine.set_mode(mode);

                const Tensor signal = tensor_from<T>(engine, c.dims, {1, 2, 3, 4, 5});
                const Tensor taps = tensor_from<T>(engine, {2}, {10, 1});

                // out[n] = 10 * signal[n + 1] + 1 * signal[n].
                EXPECT_EQ(widened(convolution_of<T>(engine, c, signal, taps)),
                          (std::vector<double>{21, 32, 43, 54}));
            }
        }

        /**
         * @brief Tests that integer data gives the formula's sums exactly in every shape and mode.
         */
        TYPED_TEST(Medium_SignalConvolve1DTyped, IntegerDataEqualsTheFormulaExactly)
        {
            using T = TypeParam;
            Engine& engine = *this->engine;

            for (const ConvolutionCase& c : table_cases())
            {
                const std::vector<double> x = residue_pattern(element_count(c.dims), 37, 23, 11);
                const std::vector<double> w =
                    residue_pattern(static_cast<std::size_t>(c.taps), 13, 7, 3);
                const std::vector<double> expected = convolution_by_formula(c, x, w);

                for (const double truth : expected)
                    ASSERT_LT(std::fabs(truth), K_HALF_EXACT_INTEGER_LIMIT) << c.id;

                for (const ExecutionMode mode : K_FORCED_MODES)
                {
                    SCOPED_TRACE(trace_of(c, mode));
                    engine.set_mode(mode);

                    const Tensor signal = tensor_from<T>(engine, c.dims, x);
                    const Tensor taps = tensor_from<T>(engine, {c.taps}, w);

                    EXPECT_EQ(widened(convolution_of<T>(engine, c, signal, taps)), expected);
                }
            }
        }

        /**
         * @brief Tests that non-integer data stays within the rounding bound of its exact sum,
         *        in every shape and mode.
         */
        TYPED_TEST(Medium_SignalConvolve1DTyped, FractionalDataStaysWithinTheRoundingBound)
        {
            using T = TypeParam;
            Engine& engine = *this->engine;

            for (const ConvolutionCase& c : table_cases())
            {
                // The library sees the rounded inputs, so the exact sum is formed from those.
                const std::vector<double> x = widened(elements_from<T>(
                    fractional(residue_pattern(element_count(c.dims), 37, 23, 11))));
                const std::vector<double> w = widened(elements_from<T>(
                    fractional(residue_pattern(static_cast<std::size_t>(c.taps), 13, 7, 3))));
                const int64_t length = c.dims.back();
                const int64_t out_length = length - c.taps + 1;

                for (const ExecutionMode mode : K_FORCED_MODES)
                {
                    SCOPED_TRACE(trace_of(c, mode));
                    engine.set_mode(mode);

                    const Tensor signal = tensor_from<T>(engine, c.dims, x);
                    const Tensor taps = tensor_from<T>(engine, {c.taps}, w);
                    const std::vector<double> got =
                        widened(convolution_of<T>(engine, c, signal, taps));

                    for (int64_t row = 0; row < c.rows(); ++row)
                    {
                        for (int64_t n = 0; n < out_length; ++n)
                        {
                            TruthSum sum;
                            for (int64_t k = 0; k < c.taps; ++k)
                                sum.add_product(
                                    x[static_cast<std::size_t>(row * length + n + c.taps - 1 - k)],
                                    w[static_cast<std::size_t>(k)]);
                            const TruthValue truth = sum.value();

                            EXPECT_NEAR(got[static_cast<std::size_t>(row * out_length + n)],
                                        truth.value, convolution_bound<T>(mode, c.taps, truth))
                                << "row " << row << " sample " << n;
                        }
                    }
                }
            }
        }

        /** @brief Tests that one device run repeats to the bit on non-integer data. */
        TYPED_TEST(Medium_SignalConvolve1DTyped, TheDeviceRepeatsToTheBit)
        {
            using T = TypeParam;
            Engine& engine = *this->engine;
            engine.set_mode(ExecutionMode::ForceSYCL);

            const ConvolutionCase c{"repeat", {5, 64}, 17};
            const std::vector<double> x =
                fractional(residue_pattern(element_count(c.dims), 37, 23, 11));
            const std::vector<double> w =
                fractional(residue_pattern(static_cast<std::size_t>(c.taps), 13, 7, 3));
            const Tensor signal = tensor_from<T>(engine, c.dims, x);
            const Tensor taps = tensor_from<T>(engine, {c.taps}, w);

            const std::vector<T> first = convolution_of<T>(engine, c, signal, taps);
            const std::vector<T> second = convolution_of<T>(engine, c, signal, taps);

            ASSERT_EQ(first.size(), second.size());
            EXPECT_EQ(std::memcmp(first.data(), second.data(), first.size() * sizeof(T)), 0);
        }

        /** @brief Tests that the operation refuses every malformed operand at record time. */
        TEST_F(Medium_SignalConvolve1D, RefusesMismatchedOperandsAtRecordTime)
        {
            RecordingFixture graph(*engine);
            SignalOps ops(graph.recorder());

            const Tensor signal = tensor_from<float>(*engine, {2, 8}, std::vector<double>(16, 1.0));
            const Tensor taps = tensor_from<float>(*engine, {3}, {1, 1, 1});
            Tensor out = sentinel_tensor<float>(*engine, {2, 6});

            // The control: this call is well-formed and records.
            EXPECT_NO_THROW(ops.convolve_1d(signal, taps, out));
            graph.run().value();

            const Tensor double_taps = tensor_from<double>(*engine, {3}, {1, 1, 1});
            expect_refusal([&] { ops.convolve_1d(signal, double_taps, out); }, "taps");

            Tensor double_out = sentinel_tensor<double>(*engine, {2, 6});
            expect_refusal([&] { ops.convolve_1d(signal, taps, double_out); }, "out");

            const Tensor matrix_taps = tensor_from<float>(*engine, {1, 3}, {1, 1, 1});
            expect_refusal([&] { ops.convolve_1d(signal, matrix_taps, out); }, "taps");

            Tensor flat_out = sentinel_tensor<float>(*engine, {12});
            expect_refusal([&] { ops.convolve_1d(signal, taps, flat_out); }, "out");

            Tensor wrong_rows = sentinel_tensor<float>(*engine, {3, 6});
            expect_refusal([&] { ops.convolve_1d(signal, taps, wrong_rows); }, "out");

            Tensor wrong_length = sentinel_tensor<float>(*engine, {2, 7});
            expect_refusal([&] { ops.convolve_1d(signal, taps, wrong_length); }, "out");

            const Tensor no_taps = tensor_from<float>(*engine, {0}, {});
            Tensor full_length = sentinel_tensor<float>(*engine, {2, 9});
            expect_refusal([&] { ops.convolve_1d(signal, no_taps, full_length); }, "taps");

            const Tensor long_taps =
                tensor_from<float>(*engine, {9}, std::vector<double>(9, 1.0));
            Tensor no_room = sentinel_tensor<float>(*engine, {2, 0});
            expect_refusal([&] { ops.convolve_1d(signal, long_taps, no_room); }, "signal");
        }

        /**
         * @brief Tests that a call with options puts the name and the class on the node, and one
         *        without keeps the operation id and Unstated.
         */
        TEST_F(Medium_SignalConvolve1D, TheNodeCarriesTheNameAndTheDeterminismClass)
        {
            SignalOps ops(recorder());
            const Tensor signal = tensor_from<float>(*engine, {2, 8}, std::vector<double>(16, 1.0));
            const Tensor taps = tensor_from<float>(*engine, {3}, {1, 1, 1});
            Tensor stated = sentinel_tensor<float>(*engine, {2, 6});
            Tensor silent = sentinel_tensor<float>(*engine, {2, 6});

            ops.convolve_1d(signal, taps, stated,
                            OperationOptions{"audio_fir", SushiRuntime::Determinism::Bitwise});
            ops.convolve_1d(signal, taps, silent);

            SushiRuntime::Result<SushiRuntime::PlanView> plan = recording->plan();
            ASSERT_TRUE(plan);
            ASSERT_EQ(plan->nodes.size(), std::size_t{2});
            EXPECT_EQ(plan->nodes[0].name, "audio_fir");
            EXPECT_EQ(plan->nodes[0].determinism, SushiRuntime::Determinism::Bitwise);
            EXPECT_EQ(plan->nodes[1].name, "signal.convolve_1d");
            EXPECT_EQ(plan->nodes[1].determinism, SushiRuntime::Determinism::Unstated);
        }

        /**
         * @brief Tests that an empty name and a class outside the enum are refused at record time.
         */
        TEST_F(Medium_SignalConvolve1D, AnEmptyNameOrAnUnknownClassIsRefused)
        {
            SignalOps ops(recorder());
            const Tensor signal = tensor_from<float>(*engine, {2, 8}, std::vector<double>(16, 1.0));
            const Tensor taps = tensor_from<float>(*engine, {3}, {1, 1, 1});
            Tensor out = sentinel_tensor<float>(*engine, {2, 6});

            EXPECT_THROW(ops.convolve_1d(signal, taps, out,
                                         OperationOptions{"", SushiRuntime::Determinism::Bitwise}),
                         std::invalid_argument);
            EXPECT_THROW(ops.convolve_1d(signal, taps, out,
                                         OperationOptions{std::string_view(),
                                                      SushiRuntime::Determinism::Bitwise}),
                         std::invalid_argument);
            EXPECT_THROW(ops.convolve_1d(
                             signal, taps, out,
                             OperationOptions{"fir", static_cast<SushiRuntime::Determinism>(200)}),
                         std::invalid_argument);
        }

        /** @brief Tests that the operation refuses an output that is also an input. */
        TEST_F(Medium_SignalConvolve1D, RefusesAnOutputThatOverlapsAnInput)
        {
            RecordingFixture graph(*engine);
            SignalOps ops(graph.recorder());

            Tensor signal = tensor_from<float>(*engine, {2, 8}, std::vector<double>(16, 1.0));
            const Tensor one_tap = tensor_from<float>(*engine, {1}, {1});

            // One tap leaves the output shaped like the signal; one tensor passes every shape check.
            expect_refusal([&] { ops.convolve_1d(signal, one_tap, signal); }, "out");
        }
    } // namespace Test
} // namespace SushiBLAS
