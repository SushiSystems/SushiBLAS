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
 * @file test_fft.cpp
 * @brief Tests the reference batched FFT against a direct DFT, and its inverse round trip.
 * @author Mustafa Garip
 */

#include <SushiBLAS/signal_kernels/reference/fft.hpp>

#include <cmath>
#include <limits>
#include <vector>
#include <cstddef>
#include <gtest/gtest.h>
#include <SushiBLAS/signal_kernels/fft_plan.hpp>
#include <SushiBLAS/typed_suite_naming/default_type_names.hpp>

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
            };

            /** @brief Returns radix-2, Bluestein, 2-D and batched shapes. */
            std::vector<FftCase> cases()
            {
                return {{"radix-2 line", 1, 16, 1}, {"Bluestein line", 1, 12, 1},
                        {"prime line", 1, 7, 2},    {"one sample", 1, 1, 3},
                        {"2-D mixed", 6, 8, 2},     {"2-D Bluestein", 5, 9, 1}};
            }

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

            /** @brief Returns the tolerance of a transform of values bounded by 1 in T. */
            template<typename T>
            double tolerance(const FftCase& c)
            {
                return 256.0 * std::numeric_limits<T>::epsilon() *
                       static_cast<double>(c.rows * c.columns);
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
        } // namespace

        /** @brief Runs the reference transform in one scalar type. */
        template<typename T>
        class Small_ReferenceFft : public ::testing::Test
        {

        };

        using FftScalars = ::testing::Types<float, double>;
        SB_TYPED_TEST_SUITE(Small_ReferenceFft, FftScalars);

        TYPED_TEST(Small_ReferenceFft, ComplexForwardMatchesTheDirectDFT)
        {
            using T = TypeParam;
            for (const FftCase& c : cases())
            {
                SCOPED_TRACE(c.id);
                const std::size_t scalars = 2 * c.batch * c.rows * c.columns;
                const std::vector<T> input = rounded<T>(values(scalars, 1));
                std::vector<T> output(scalars, T(7));
                const Kernels::FftPlan<T> plan(c.rows, c.columns,
                                               Kernels::FftPairing::ComplexToComplex);

                Kernels::Reference::fft_forward(plan, input.data(), output.data(), c.batch);

                const std::vector<double> expected = direct_dft(widened(input), c, -1);
                for (std::size_t i = 0; i < scalars; ++i)
                    EXPECT_NEAR(output[i], expected[i], tolerance<T>(c)) << i;
            }
        }

        TYPED_TEST(Small_ReferenceFft, ComplexInverseUndoesTheForwardScaledByTheSize)
        {
            using T = TypeParam;
            for (const FftCase& c : cases())
            {
                SCOPED_TRACE(c.id);
                const std::size_t scalars = 2 * c.batch * c.rows * c.columns;
                const std::vector<T> input = rounded<T>(values(scalars, 2));
                std::vector<T> spectrum(scalars);
                std::vector<T> back(scalars, T(7));
                const Kernels::FftPlan<T> plan(c.rows, c.columns,
                                               Kernels::FftPairing::ComplexToComplex);

                Kernels::Reference::fft_inverse(plan, input.data(), spectrum.data(), c.batch);
                const std::vector<double> expected = direct_dft(widened(input), c, +1);
                const double size = static_cast<double>(c.rows * c.columns);
                for (std::size_t i = 0; i < scalars; ++i)
                    EXPECT_NEAR(spectrum[i], expected[i] / size, tolerance<T>(c)) << i;

                Kernels::Reference::fft_forward(plan, spectrum.data(), back.data(), c.batch);
                for (std::size_t i = 0; i < scalars; ++i)
                    EXPECT_NEAR(back[i], input[i], tolerance<T>(c)) << i;
            }
        }

        TYPED_TEST(Small_ReferenceFft, RealForwardMatchesTheHalfOfTheDirectDFT)
        {
            using T = TypeParam;
            for (const FftCase& c : cases())
            {
                SCOPED_TRACE(c.id);
                const std::size_t samples = c.batch * c.rows * c.columns;
                const std::size_t width = c.columns / 2 + 1;
                const std::vector<T> input = rounded<T>(values(samples, 3));
                std::vector<T> output(2 * c.batch * c.rows * width, T(7));
                const Kernels::FftPlan<T> plan(c.rows, c.columns,
                                               Kernels::FftPairing::RealToComplex);

                Kernels::Reference::fft_forward(plan, input.data(), output.data(), c.batch);

                std::vector<double> complex(2 * samples, 0.0);
                for (std::size_t i = 0; i < samples; ++i)
                    complex[2 * i] = input[i];
                const std::vector<double> full = direct_dft(complex, c, -1);
                for (std::size_t line = 0; line < c.batch * c.rows; ++line)
                    for (std::size_t k = 0; k < width; ++k)
                        for (std::size_t part = 0; part < 2; ++part)
                            EXPECT_NEAR(output[2 * (line * width + k) + part],
                                        full[2 * (line * c.columns + k) + part], tolerance<T>(c))
                                << line << " " << k;
            }
        }

        TYPED_TEST(Small_ReferenceFft, RealInverseRecoversTheSamples)
        {
            using T = TypeParam;
            for (const FftCase& c : cases())
            {
                SCOPED_TRACE(c.id);
                const std::size_t samples = c.batch * c.rows * c.columns;
                const std::vector<T> input = rounded<T>(values(samples, 4));
                std::vector<T> spectrum(2 * c.batch * c.rows * (c.columns / 2 + 1));
                std::vector<T> back(samples, T(7));
                const Kernels::FftPlan<T> plan(c.rows, c.columns,
                                               Kernels::FftPairing::RealToComplex);

                Kernels::Reference::fft_forward(plan, input.data(), spectrum.data(), c.batch);
                Kernels::Reference::fft_inverse(plan, spectrum.data(), back.data(), c.batch);

                for (std::size_t i = 0; i < samples; ++i)
                    EXPECT_NEAR(back[i], input[i], tolerance<T>(c)) << i;
            }
        }

        TEST(Small_ReferenceFftHand, AnImpulseHasAFlatSpectrum)
        {
            const std::vector<double> input{1, 0, 0, 0, 0, 0, 0, 0, 0, 0};
            std::vector<double> output(10);
            const Kernels::FftPlan<double> plan(1, 5, Kernels::FftPairing::ComplexToComplex);

            Kernels::Reference::fft_forward(plan, input.data(), output.data(), 1);

            for (std::size_t k = 0; k < 5; ++k)
            {
                EXPECT_NEAR(output[2 * k], 1.0, 1e-14) << k;
                EXPECT_NEAR(output[2 * k + 1], 0.0, 1e-14) << k;
            }
        }
    } // namespace Test
} // namespace SushiBLAS
