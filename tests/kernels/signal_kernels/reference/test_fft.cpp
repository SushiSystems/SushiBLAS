/**************************************************************************/
/* test_fft.cpp                                                           */
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
 * @file test_fft.cpp
 * @brief Tests the reference batched FFT against a direct DFT, and its inverse round trip.
 * @author Mustafa Garip
 */

#include <SushiBLAS/signal_kernels/reference/fft.hpp>

#include <cmath>
#include <string>
#include <utility>
#include <vector>
#include <cstddef>
#include <gtest/gtest.h>
#include <SushiBLAS/signal_kernels/fft_plan.hpp>
#include "support/tolerance.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief Holds one transform shape a case runs. */
            struct FftCase
            {
                    const char* id = "";     /**< The case's name in a failure message. */
                    std::size_t rows = 1;    /**< Rows per transform. */
                    std::size_t columns = 0; /**< Samples per row. */
                    std::size_t batch = 1;   /**< Transforms in the batch. */
                    bool impulse = false;    /**< Whether the signal is a unit impulse instead of pseudo-random values. */
            };

            /**
             * @brief Holds the products a transform sum is allowed per sample.
             *
             * A Bluestein line pads to a power of two of at most four times its length and runs
             * three transforms, so a sample is the sum of a few times its length of products.
             */
            constexpr std::size_t K_TERMS_PER_SAMPLE = 8;

            /** @brief Returns the samples of one transform. */
            std::size_t samples_of(const FftCase& c)
            {
                return c.rows * c.columns;
            }

            /** @brief Returns the products a transform sum holds, the length its error bound grows with. */
            std::size_t terms_of(const FftCase& c)
            {
                return K_TERMS_PER_SAMPLE * samples_of(c);
            }

            /** @brief Returns count fixed pseudo-random values in [-1, 1) from a seed. */
            std::vector<double> pseudo_random_values(std::size_t count, unsigned seed)
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

            /** @brief Returns count scalars of the case's signal: a unit impulse or pseudo-random values. */
            std::vector<double> signal_of(const FftCase& c, std::size_t count, unsigned seed)
            {
                if (!c.impulse)
                    return pseudo_random_values(count, seed);
                std::vector<double> out(count, 0.0);
                out[0] = 1.0;
                return out;
            }

            /**
             * @brief Returns the direct DFT of interleaved complex entries, in long double.
             * @param sign -1 for the forward transform, +1 for the unscaled inverse.
             */
            std::vector<double> direct_dft(const std::vector<double>& in, const FftCase& c,
                                           int sign)
            {
                const long double pi = 3.141592653589793238462643383279502884L;
                const std::size_t entry = c.rows * c.columns;
                std::vector<double> out(in.size());
                for (std::size_t b = 0; b < c.batch; ++b)
                    for (std::size_t u = 0; u < c.rows; ++u)
                        for (std::size_t v = 0; v < c.columns; ++v)
                        {
                            long double re = 0.0L;
                            long double im = 0.0L;
                            for (std::size_t r = 0; r < c.rows; ++r)
                                for (std::size_t k = 0; k < c.columns; ++k)
                                {
                                    const long double angle =
                                        sign * 2.0L * pi *
                                        (static_cast<long double>((u * r) % c.rows) / c.rows +
                                         static_cast<long double>((v * k) % c.columns) / c.columns);
                                    const std::size_t at = b * entry + r * c.columns + k;
                                    const long double xr = in[2 * at];
                                    const long double xi = in[2 * at + 1];
                                    re += xr * std::cos(angle) - xi * std::sin(angle);
                                    im += xr * std::sin(angle) + xi * std::cos(angle);
                                }
                            const std::size_t at = b * entry + u * c.columns + v;
                            out[2 * at] = static_cast<double>(re);
                            out[2 * at + 1] = static_cast<double>(im);
                        }
                return out;
            }

            /** @brief Returns the values divided by a divisor. */
            std::vector<double> divided(std::vector<double> values, double divisor)
            {
                for (double& value : values)
                    value /= divisor;
                return values;
            }

            /** @brief Returns the bins 0 to columns / 2 of every line of a full complex spectrum, the half a real transform keeps. */
            std::vector<double> half_of(const std::vector<double>& full, const FftCase& c)
            {
                const std::size_t width = c.columns / 2 + 1;
                std::vector<double> half;
                for (std::size_t line = 0; line < c.batch * c.rows; ++line)
                    for (std::size_t k = 0; k < width; ++k)
                        for (std::size_t part = 0; part < 2; ++part)
                            half.push_back(full[2 * (line * c.columns + k) + part]);
                return half;
            }

            /** @brief Returns values rounded to T. */
            template<typename T>
            std::vector<T> rounded(const std::vector<double>& in)
            {
                return std::vector<T>(in.begin(), in.end());
            }

            /** @brief Returns values widened to double. */
            template<typename T>
            std::vector<double> widened(const std::vector<T>& in)
            {
                return std::vector<double>(in.begin(), in.end());
            }

            /** @brief Holds what a transform wrote, what the oracle gives and the bound the pair is held to. */
            struct FftOutcome
            {
                    std::vector<double> computed; /**< The scalars the transform wrote, widened to double. */
                    std::vector<double> expected; /**< The scalars the oracle gives. */
                    double magnitude = 1.0;       /**< The largest sum of absolute products one scalar can hold. */
                    double roundoff = 0.0;        /**< The unit roundoff of the scalar type the transform ran in. */
            };

            /** @brief Returns the outcome of a transform run in T. */
            template<typename T>
            FftOutcome outcome_in(const std::vector<T>& computed, std::vector<double> expected,
                                  double magnitude)
            {
                return {widened(computed), std::move(expected), magnitude,
                        TestSupport::unit_roundoff<T>()};
            }

            /** @brief Expects each computed scalar to match the expected one within the rounding bound of a transform sum. */
            void expect_within_rounding(const FftOutcome& outcome, const FftCase& c)
            {
                ASSERT_EQ(outcome.computed.size(), outcome.expected.size());
                const double bound = static_cast<double>(terms_of(c)) * outcome.roundoff * outcome.magnitude;
                for (std::size_t i = 0; i < outcome.expected.size(); ++i)
                    EXPECT_TRUE(TestSupport::is_within(
                        TestSupport::measure_error(outcome.computed[i], outcome.expected[i], bound)))
                        << i;
            }

            /** @brief Runs the complex forward transform and gives the direct DFT; its values reach the number of samples. */
            template<typename T>
            FftOutcome outcome_complex_forward(const FftCase& c)
            {
                const std::size_t scalars = 2 * c.batch * samples_of(c);
                const std::vector<T> input = rounded<T>(signal_of(c, scalars, 1));
                std::vector<T> output(scalars, T(7));
                const Kernels::FftPlan<T> plan(c.rows, c.columns,
                                               Kernels::FftPairing::ComplexToComplex);

                Kernels::Reference::fft_forward(plan, input.data(), output.data(), c.batch);

                return outcome_in(output, direct_dft(widened(input), c, -1),
                                  static_cast<double>(samples_of(c)));
            }

            /** @brief Runs the complex inverse and gives the direct unscaled DFT divided by the number of samples. */
            template<typename T>
            FftOutcome outcome_complex_inverse(const FftCase& c)
            {
                const std::size_t scalars = 2 * c.batch * samples_of(c);
                const std::vector<T> input = rounded<T>(signal_of(c, scalars, 2));
                std::vector<T> spectrum(scalars, T(7));
                const Kernels::FftPlan<T> plan(c.rows, c.columns,
                                               Kernels::FftPairing::ComplexToComplex);

                Kernels::Reference::fft_inverse(plan, input.data(), spectrum.data(), c.batch);

                return outcome_in(spectrum,
                                  divided(direct_dft(widened(input), c, +1),
                                          static_cast<double>(samples_of(c))),
                                  1.0);
            }

            /** @brief Runs the complex inverse then the forward transform and gives the input. */
            template<typename T>
            FftOutcome outcome_complex_round_trip(const FftCase& c)
            {
                const std::size_t scalars = 2 * c.batch * samples_of(c);
                const std::vector<T> input = rounded<T>(signal_of(c, scalars, 2));
                std::vector<T> spectrum(scalars);
                std::vector<T> back(scalars, T(7));
                const Kernels::FftPlan<T> plan(c.rows, c.columns,
                                               Kernels::FftPairing::ComplexToComplex);

                Kernels::Reference::fft_inverse(plan, input.data(), spectrum.data(), c.batch);
                Kernels::Reference::fft_forward(plan, spectrum.data(), back.data(), c.batch);

                return outcome_in(back, widened(input), 1.0);
            }

            /** @brief Runs the real forward transform and gives the kept half of the direct DFT. */
            template<typename T>
            FftOutcome outcome_real_forward(const FftCase& c)
            {
                const std::size_t samples = c.batch * samples_of(c);
                const std::vector<T> input = rounded<T>(signal_of(c, samples, 3));
                std::vector<T> output(2 * c.batch * c.rows * (c.columns / 2 + 1), T(7));
                const Kernels::FftPlan<T> plan(c.rows, c.columns,
                                               Kernels::FftPairing::RealToComplex);

                Kernels::Reference::fft_forward(plan, input.data(), output.data(), c.batch);

                std::vector<double> complex(2 * samples, 0.0);
                for (std::size_t i = 0; i < samples; ++i)
                    complex[2 * i] = input[i];
                return outcome_in(output, half_of(direct_dft(complex, c, -1), c),
                                  static_cast<double>(samples_of(c)));
            }

            /** @brief Runs the real forward then the real inverse transform and gives the input. */
            template<typename T>
            FftOutcome outcome_real_round_trip(const FftCase& c)
            {
                const std::size_t samples = c.batch * samples_of(c);
                const std::vector<T> input = rounded<T>(signal_of(c, samples, 4));
                std::vector<T> spectrum(2 * c.batch * c.rows * (c.columns / 2 + 1));
                std::vector<T> back(samples, T(7));
                const Kernels::FftPlan<T> plan(c.rows, c.columns,
                                               Kernels::FftPairing::RealToComplex);

                Kernels::Reference::fft_forward(plan, input.data(), spectrum.data(), c.batch);
                Kernels::Reference::fft_inverse(plan, spectrum.data(), back.data(), c.batch);

                return outcome_in(back, widened(input), 1.0);
            }

            /** @brief Holds the runs of one scalar type. */
            struct FftRuns
            {
                    FftOutcome (*complex_forward)(const FftCase&);
                    FftOutcome (*complex_inverse)(const FftCase&);
                    FftOutcome (*complex_round_trip)(const FftCase&);
                    FftOutcome (*real_forward)(const FftCase&);
                    FftOutcome (*real_round_trip)(const FftCase&);
            };

            /** @brief Returns the runs in T. */
            template<typename T>
            constexpr FftRuns runs_in()
            {
                return {&outcome_complex_forward<T>, &outcome_complex_inverse<T>,
                        &outcome_complex_round_trip<T>, &outcome_real_forward<T>,
                        &outcome_real_round_trip<T>};
            }

            /** @brief One case in one scalar type. */
            struct FftRow
            {
                    std::string name;
                    FftCase shape;
                    FftRuns runs;
            };

            /** @brief Returns radix-2, Bluestein, 2-D, batched and impulse shapes. */
            std::vector<FftCase> cases()
            {
                return {{"Radix2Line", 1, 16, 1, false}, {"BluesteinLine", 1, 12, 1, false},
                        {"PrimeLine", 1, 7, 2, false},   {"OneSample", 1, 1, 3, false},
                        {"TwoDMixed", 6, 8, 2, false},   {"TwoDBluestein", 5, 9, 1, false},
                        {"Impulse", 1, 5, 1, true}};
            }

            /** @brief Returns every case in float and in double. */
            std::vector<FftRow> fft_rows()
            {
                std::vector<FftRow> rows;
                for (const FftCase& c : cases())
                {
                    rows.push_back({std::string(c.id) + "_Float", c, runs_in<float>()});
                    rows.push_back({std::string(c.id) + "_Double", c, runs_in<double>()});
                }
                return rows;
            }

            /** @brief Names a row, which gtest takes as the test's suffix. */
            std::string fft_row_name(const ::testing::TestParamInfo<FftRow>& info)
            {
                return info.param.name;
            }
        } // namespace

        /** @brief Runs the reference transform on one case in one scalar type. */
        class Small_ReferenceFft : public ::testing::TestWithParam<FftRow>
        {

        };

        TEST_P(Small_ReferenceFft, ComplexForwardMatchesTheDirectDFT)
        {
            expect_within_rounding(GetParam().runs.complex_forward(GetParam().shape), GetParam().shape);
        }

        TEST_P(Small_ReferenceFft, ComplexInverseMatchesTheDirectDFTScaledByTheSize)
        {
            expect_within_rounding(GetParam().runs.complex_inverse(GetParam().shape), GetParam().shape);
        }

        TEST_P(Small_ReferenceFft, ComplexForwardUndoesTheInverse)
        {
            expect_within_rounding(GetParam().runs.complex_round_trip(GetParam().shape), GetParam().shape);
        }

        TEST_P(Small_ReferenceFft, RealForwardMatchesTheHalfOfTheDirectDFT)
        {
            expect_within_rounding(GetParam().runs.real_forward(GetParam().shape), GetParam().shape);
        }

        TEST_P(Small_ReferenceFft, RealInverseRecoversTheSamples)
        {
            expect_within_rounding(GetParam().runs.real_round_trip(GetParam().shape), GetParam().shape);
        }

        INSTANTIATE_TEST_SUITE_P(, Small_ReferenceFft, ::testing::ValuesIn(fft_rows()), fft_row_name);
    } // namespace Test
} // namespace SushiBLAS
