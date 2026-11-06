/**************************************************************************/
/* test_transforms.cpp                                                    */
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
 * @file test_transforms.cpp
 * @brief Tests TransformsOps::fft and inverse_fft in both forced modes against a direct DFT, and
 * their refusals.
 * @author Mustafa Garip
 */

#include <SushiBLAS/experimental/signal/transforms.hpp>

#include <cmath>
#include <string>
#include <vector>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/SushiBLAS.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/tensor_expectations/scalar_dtype.hpp>
#include <SushiBLAS/typed_suite_naming/default_type_names.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>
#include "support/tolerance.hpp"
namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /// @brief The forced modes every case runs in.
            constexpr ExecutionMode K_FORCED_MODES[] = {ExecutionMode::ForceSYCL,
                                                        ExecutionMode::ForceReference};

            /** @brief Returns count fixed pseudo-random values in [-1, 1) from a seed. */
            std::vector<double> values(std::size_t count, unsigned seed)
            {
                std::vector<double> out(count);
                unsigned state = seed * 2654435761u + 1u;
                for (double& v : out)
                {
                    state = state * 1664525u + 1013904223u;
                    v = static_cast<double>(state >> 8) / static_cast<double>(1u << 24) * 2.0 - 1.0;
                }
                return out;
            }

            /** @brief Returns the direct forward DFT of the entries of shape, in long double. */
            std::vector<std::complex<double>>
            direct_dft(const std::vector<std::complex<double>>& in, const FftShape& shape)
            {
                const long double pi = 3.141592653589793238462643383279502884L;
                const std::size_t entry = shape.rows * shape.columns;
                std::vector<std::complex<double>> out(in.size());
                for (std::size_t b = 0; b < shape.batch; ++b)
                    for (std::size_t u = 0; u < shape.rows; ++u)
                        for (std::size_t v = 0; v < shape.columns; ++v)
                        {
                            long double re = 0.0L;
                            long double im = 0.0L;
                            for (std::size_t r = 0; r < shape.rows; ++r)
                                for (std::size_t k = 0; k < shape.columns; ++k)
                                {
                                    const long double angle =
                                        -2.0L * pi *
                                        (static_cast<long double>((u * r) % shape.rows) /
                                             shape.rows +
                                         static_cast<long double>((v * k) % shape.columns) /
                                             shape.columns);
                                    const std::complex<double> x =
                                        in[b * entry + r * shape.columns + k];
                                    re += x.real() * std::cos(angle) - x.imag() * std::sin(angle);
                                    im += x.real() * std::sin(angle) + x.imag() * std::cos(angle);
                                }
                            out[b * entry + u * shape.columns + v] = {static_cast<double>(re),
                                                                      static_cast<double>(im)};
                        }
                return out;
            }

            /** @brief Returns the sum of |x| over the entry `batch` of the inputs of shape. */
            double entry_magnitude(const std::vector<std::complex<double>>& in, const FftShape& shape,
                                   std::size_t batch)
            {
                const std::size_t entry = shape.rows * shape.columns;
                double sum = 0.0;
                for (std::size_t i = 0; i < entry; ++i)
                    sum += std::abs(in[batch * entry + i]);
                return sum;
            }

            /** @brief Returns a rank-1 tensor holding elements. */
            template<typename E>
            Tensor tensor_of(Engine& engine, const std::vector<E>& elements)
            {
                Tensor t = engine.create_tensor({static_cast<int64_t>(elements.size())},
                                                sushiblas_dtype<E>());
                SushiBLAS::IO().from_vector(t, elements);
                return t;
            }

            /** @brief Returns true when the test device can compute in T. */
            template<typename T>
            bool device_computes()
            {
                if constexpr (std::is_same_v<T, double>)
                    return TestEnvironment::queue().get_device().has(sycl::aspect::fp64);
                else
                    return true;
            }

            /** @brief Returns the 1-D, Bluestein, 2-D and batched shapes every mode runs. */
            std::vector<FftShape> shapes(FftKind kind)
            {
                return {{1, 16, 1, kind}, {1, 12, 2, kind}, {4, 6, 2, kind}, {3, 7, 1, kind}};
            }

            /** @brief Returns the name a failure message gives a shape run in a mode. */
            std::string trace_of(const FftShape& shape, ExecutionMode mode)
            {
                return std::to_string(shape.rows) + "x" + std::to_string(shape.columns) +
                       (mode == ExecutionMode::ForceReference ? " ForceReference" : " ForceSYCL");
            }

            /** @brief Runs check once for each forced mode. */
            template<typename Check>
            void in_each_forced_mode(Check&& check)
            {
                for (const ExecutionMode mode : K_FORCED_MODES)
                    check(mode);
            }

            /**
             * @brief Expects record to throw std::invalid_argument naming op and operand.
             * @tparam Record Callable `() -> void` that records one malformed call.
             */
            template<typename Record>
            void expect_refusal(Record&& record, const char* op, const char* operand)
            {
                try
                {
                    record();
                    ADD_FAILURE() << op << " accepted a malformed '" << operand << "'";
                }
                catch (const std::invalid_argument& error)
                {
                    const std::string text = error.what();
                    EXPECT_NE(text.find(op), std::string::npos) << text;
                    EXPECT_NE(text.find("'" + std::string(operand) + "'"), std::string::npos)
                        << text;
                }
            }

            /**
             * @brief Expects a computed value to equal the exact one within n·u·sum|x|.
             * @param magnitude The sum of the magnitudes of the inputs of the entry.
             * @param shape The shape; the transform sums rows * columns products per value.
             */
            template<typename T>
            void expect_transform_value(double computed, double exact, double magnitude,
                                        const FftShape& shape)
            {
                EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_sum_error<T>(
                    computed, exact, magnitude, shape.rows * shape.columns)));
            }

            /** @brief Expects the complex forward transform of shape to match the direct DFT in mode. */
            template<typename T>
            void expect_complex_forward_of(Engine& engine, const FftShape& shape, ExecutionMode mode)
            {
                using C = std::complex<T>;
                SCOPED_TRACE(trace_of(shape, mode));
                engine.set_mode(mode);

                const std::size_t count = shape.batch * shape.rows * shape.columns;
                const std::vector<double> raw = values(2 * count, 51);
                std::vector<C> input(count);
                std::vector<std::complex<double>> exact(count);
                for (std::size_t i = 0; i < count; ++i)
                {
                    input[i] = C(static_cast<T>(raw[2 * i]), static_cast<T>(raw[2 * i + 1]));
                    exact[i] = {input[i].real(), input[i].imag()};
                }
                const std::vector<std::complex<double>> expected = direct_dft(exact, shape);

                const Tensor in = tensor_of(engine, input);
                Tensor out = tensor_of(engine, std::vector<C>(count, C(7)));
                RecordingFixture graph(engine);
                TransformsOps(graph.recorder()).fft(in, out, shape);
                graph.run().value();

                const std::vector<C> got = SushiBLAS::IO().to_vector<C>(out);
                const std::size_t entry = shape.rows * shape.columns;
                for (std::size_t i = 0; i < count; ++i)
                {
                    const double magnitude = entry_magnitude(exact, shape, i / entry);
                    expect_transform_value<T>(got[i].real(), expected[i].real(), magnitude, shape);
                    expect_transform_value<T>(got[i].imag(), expected[i].imag(), magnitude, shape);
                }
            }

            /** @brief Expects the complex forward transform to match the direct DFT in every shape in mode. */
            template<typename T>
            void expect_complex_forward(Engine& engine, ExecutionMode mode)
            {
                for (const FftShape& shape : shapes(FftKind::ComplexToComplex))
                    expect_complex_forward_of<T>(engine, shape, mode);
            }

            /** @brief Holds what the real transform and its inverse wrote for one shape. */
            template<typename T>
            struct RealRun
            {
                std::vector<T> input;                      /**< The real samples. */
                std::vector<std::complex<double>> exact;   /**< The samples as complex doubles. */
                std::vector<std::complex<T>> half_spectrum; /**< The half spectrum fft wrote. */
                std::vector<T> samples;                    /**< The samples inverse_fft recovered. */
            };

            /** @brief Runs fft then inverse_fft over the real samples of shape in mode. */
            template<typename T>
            RealRun<T> run_real(Engine& engine, const FftShape& shape, ExecutionMode mode)
            {
                using C = std::complex<T>;
                engine.set_mode(mode);

                const std::size_t count = shape.batch * shape.rows * shape.columns;
                const std::size_t spectrum_count = shape.batch * shape.rows * (shape.columns / 2 + 1);
                const std::vector<double> raw = values(count, 52);

                RealRun<T> run;
                run.input = std::vector<T>(raw.begin(), raw.end());
                run.exact.resize(count);
                for (std::size_t i = 0; i < count; ++i)
                    run.exact[i] = {static_cast<double>(run.input[i]), 0.0};

                const Tensor in = tensor_of(engine, run.input);
                Tensor spectrum = tensor_of(engine, std::vector<C>(spectrum_count, C(7)));
                Tensor back = tensor_of(engine, std::vector<T>(count, T(7)));
                RecordingFixture graph(engine);
                TransformsOps view(graph.recorder());
                view.fft(in, spectrum, shape);
                view.inverse_fft(spectrum, back, shape);
                graph.run().value();

                run.half_spectrum = SushiBLAS::IO().to_vector<C>(spectrum);
                run.samples = SushiBLAS::IO().to_vector<T>(back);
                return run;
            }

            /** @brief Expects the half spectrum of every real shape to match the direct DFT in mode. */
            template<typename T>
            void expect_real_forward(Engine& engine, ExecutionMode mode)
            {
                for (const FftShape& shape : shapes(FftKind::RealToComplex))
                {
                    SCOPED_TRACE(trace_of(shape, mode));
                    const RealRun<T> run = run_real<T>(engine, shape, mode);
                    const std::vector<std::complex<double>> full = direct_dft(run.exact, shape);
                    const std::size_t width = shape.columns / 2 + 1;

                    for (std::size_t line = 0; line < shape.batch * shape.rows; ++line)
                    {
                        const double magnitude = entry_magnitude(run.exact, shape, line / shape.rows);
                        for (std::size_t k = 0; k < width; ++k)
                        {
                            const std::complex<T> g = run.half_spectrum[line * width + k];
                            const std::complex<double> e = full[line * shape.columns + k];
                            expect_transform_value<T>(g.real(), e.real(), magnitude, shape);
                            expect_transform_value<T>(g.imag(), e.imag(), magnitude, shape);
                        }
                    }
                }
            }

            /** @brief Expects the inverse of the real transform to recover the samples of every shape in mode. */
            template<typename T>
            void expect_real_round_trip(Engine& engine, ExecutionMode mode)
            {
                for (const FftShape& shape : shapes(FftKind::RealToComplex))
                {
                    SCOPED_TRACE(trace_of(shape, mode));
                    const RealRun<T> run = run_real<T>(engine, shape, mode);
                    const std::size_t entry = shape.rows * shape.columns;

                    for (std::size_t i = 0; i < run.input.size(); ++i)
                        expect_transform_value<T>(run.samples[i], run.input[i],
                                                  entry_magnitude(run.exact, shape, i / entry), shape);
                }
            }
        } // namespace

        /** @brief Runs the transforms in one scalar type, skipped where the device lacks it. */
        template<typename T>
        class Medium_TransformsTyped : public SushiBLASTest
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

        using TransformScalars = ::testing::Types<float, double>;
        SB_TYPED_TEST_SUITE(Medium_TransformsTyped, TransformScalars);

        /** @brief Runs the refusals, which fix one scalar type. */
        class Medium_Transforms : public SushiBLASTest
        {

        };

        TYPED_TEST(Medium_TransformsTyped, ComplexForwardMatchesTheDirectDFTInBothModes)
        {
            in_each_forced_mode([&](ExecutionMode mode)
                                { expect_complex_forward<TypeParam>(*this->engine, mode); });
        }

        /** @brief Tests that the half spectrum of a real signal matches the direct DFT. */
        TYPED_TEST(Medium_TransformsTyped, RealForwardMatchesTheDirectDFT)
        {
            in_each_forced_mode([&](ExecutionMode mode)
                                { expect_real_forward<TypeParam>(*this->engine, mode); });
        }

        /** @brief Tests that the inverse of the real transform recovers the samples. */
        TYPED_TEST(Medium_TransformsTyped, RealInverseRoundTrips)
        {
            in_each_forced_mode([&](ExecutionMode mode)
                                { expect_real_round_trip<TypeParam>(*this->engine, mode); });
        }

        TEST_F(Medium_Transforms, RefusesAMismatchedDtype)
        {
            Engine& engine = *this->engine;
            const FftShape shape{1, 4, 1, FftKind::ComplexToComplex};
            const Tensor in = tensor_of(engine, std::vector<float>(4, 0.0f));
            Tensor out = tensor_of(engine, std::vector<std::complex<float>>(4));
            RecordingFixture graph(engine);

            expect_refusal([&] { TransformsOps(graph.recorder()).fft(in, out, shape); },
                           "signal.fft", "input");
        }

        TEST_F(Medium_Transforms, RefusesAWrongElementCount)
        {
            Engine& engine = *this->engine;
            const FftShape shape{2, 4, 1, FftKind::RealToComplex};
            const Tensor in = tensor_of(engine, std::vector<float>(8, 0.0f));
            Tensor out = tensor_of(engine, std::vector<std::complex<float>>(8));
            RecordingFixture graph(engine);

            // Two rows of 4 real samples have 2 * 3 half-spectrum elements, not 8.
            expect_refusal([&] { TransformsOps(graph.recorder()).fft(in, out, shape); },
                           "signal.fft", "output");
        }

        TEST_F(Medium_Transforms, RefusesAnEmptyShape)
        {
            Engine& engine = *this->engine;
            const FftShape shape{1, 0, 1, FftKind::ComplexToComplex};
            const Tensor in = tensor_of(engine, std::vector<std::complex<float>>(1));
            Tensor out = tensor_of(engine, std::vector<std::complex<float>>(1));
            RecordingFixture graph(engine);

            expect_refusal([&] { TransformsOps(graph.recorder()).inverse_fft(in, out, shape); },
                           "signal.inverse_fft", "shape");
        }

        TEST_F(Medium_Transforms, RefusesAnOutputThatIsTheInput)
        {
            Engine& engine = *this->engine;
            const FftShape shape{1, 4, 1, FftKind::ComplexToComplex};
            Tensor both = tensor_of(engine, std::vector<std::complex<float>>(4));
            RecordingFixture graph(engine);

            expect_refusal([&] { TransformsOps(graph.recorder()).fft(both, both, shape); },
                           "signal.fft", "output");
        }
    } // namespace Test
} // namespace SushiBLAS
