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
 * @brief Tests that the SYCL batched FFT equals the reference bit for bit on the host device.
 * @author Mustafa Garip
 */

#include <SushiBLAS/signal_kernels/sycl/fft.hpp>

#include <string>
#include <vector>
#include <cstddef>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/signal_kernels/fft_plan.hpp>
#include <SushiBLAS/signal_kernels/reference/fft.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>
#include "support/tolerance.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        namespace
        {
            /** @brief Holds one transform shape and pairing a case runs. */
            struct DeviceFftCase
            {
                    const char* id = "";     /**< The case's name in a failure message. */
                    std::size_t rows = 1;    /**< Rows per transform. */
                    std::size_t columns = 0; /**< Samples per row. */
                    std::size_t batch = 1;   /**< Transforms in the batch. */
                    Kernels::FftPairing pairing = Kernels::FftPairing::ComplexToComplex;
                        /**< Element pairing. */
            };

            /**
             * @brief Holds the products a transform sum is allowed per sample.
             *
             * A Bluestein line pads to a power of two of at most four times its length and runs
             * three transforms, so a sample is the sum of a few times its length of products.
             */
            constexpr std::size_t K_TERMS_PER_SAMPLE = 8;

            /** @brief Returns radix-2, Bluestein, packed and unpacked real, 1-D and 2-D cases. */
            std::vector<DeviceFftCase> cases()
            {
                using P = Kernels::FftPairing;
                return {{"ComplexRadix2", 1, 32, 3, P::ComplexToComplex},
                        {"ComplexBluestein", 1, 10, 2, P::ComplexToComplex},
                        {"Complex2D", 4, 6, 2, P::ComplexToComplex},
                        {"RealPacked", 1, 16, 2, P::RealToComplex},
                        {"RealPackedBluestein", 3, 12, 2, P::RealToComplex},
                        {"RealOdd", 5, 7, 1, P::RealToComplex}};
            }

            /** @brief Names a case, which gtest takes as the test's suffix. */
            std::string device_fft_case_name(const ::testing::TestParamInfo<DeviceFftCase>& info)
            {
                return info.param.id;
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
             * @brief Runs one transform on the queue and returns what it wrote.
             * @param forward true for fft_forward, false for fft_inverse.
             */
            std::vector<double> run_device(sycl::queue& q, const Kernels::FftPlan<double>& plan,
                                           const std::vector<double>& input,
                                           std::size_t output_scalars, std::size_t batch,
                                           bool forward)
            {
                const SharedArray<double> device_input(q, input);
                SharedArray<double> device_output(q, output_scalars);
                SharedArray<unsigned char> scratch(q,
                                                   Kernels::SYCL::fft_scratch_bytes(plan, batch));
                const sycl::event uploaded =
                    Kernels::SYCL::upload_fft_tables(q, plan, batch, scratch.get(), {});
                const Kernels::SYCL::FftDeviceTables<double> tables =
                    Kernels::SYCL::fft_device_tables(plan, batch, scratch.get());
                if (forward)
                    Kernels::SYCL::fft_forward(q, plan, tables, device_input.get(),
                                               device_output.get(), batch, {uploaded})
                        .wait();
                else
                    Kernels::SYCL::fft_inverse(q, plan, tables, device_input.get(),
                                               device_output.get(), batch, {uploaded})
                        .wait();
                return device_output.values();
            }

            /** @brief Returns what the reference forward transform writes for the input. */
            std::vector<double> reference_forward(const Kernels::FftPlan<double>& plan,
                                                  const std::vector<double>& input, std::size_t batch)
            {
                std::vector<double> spectrum(batch * plan.spectrum_scalars());
                Kernels::Reference::fft_forward(plan, input.data(), spectrum.data(), batch);
                return spectrum;
            }

            /** @brief Returns what the reference inverse transform writes for the spectrum. */
            std::vector<double> reference_inverse(const Kernels::FftPlan<double>& plan,
                                                  const std::vector<double>& spectrum, std::size_t batch)
            {
                std::vector<double> samples(batch * plan.sample_scalars());
                Kernels::Reference::fft_inverse(plan, spectrum.data(), samples.data(), batch);
                return samples;
            }

            /** @brief Expects each recovered sample to match the original within the rounding of a transform sum. */
            void expect_within_rounding(const std::vector<double>& recovered,
                                        const std::vector<double>& original, const DeviceFftCase& c)
            {
                ASSERT_EQ(recovered.size(), original.size());
                for (std::size_t i = 0; i < original.size(); ++i)
                    EXPECT_TRUE(TestSupport::is_within(TestSupport::measure_sum_error<double>(
                        recovered[i], original[i], 1.0, K_TERMS_PER_SAMPLE * c.rows * c.columns)))
                        << i;
            }
        } // namespace

        /** @brief Runs the kernels on the host CPU queue, one case at a time. */
        class Medium_SYCLFftCases : public HostDeviceTest, public ::testing::WithParamInterface<DeviceFftCase>
        {

        };

        TEST_P(Medium_SYCLFftCases, ForwardEqualsTheReferenceBitForBit)
        {
            const DeviceFftCase& c = GetParam();
            const Kernels::FftPlan<double> plan(c.rows, c.columns, c.pairing);
            const std::vector<double> input = values(c.batch * plan.sample_scalars(), 5);
            const std::vector<double> expected = reference_forward(plan, input, c.batch);

            const std::vector<double> got =
                run_device(queue(), plan, input, expected.size(), c.batch, true);

            EXPECT_EQ(got, expected);
        }

        TEST_P(Medium_SYCLFftCases, InverseEqualsTheReferenceBitForBit)
        {
            const DeviceFftCase& c = GetParam();
            const Kernels::FftPlan<double> plan(c.rows, c.columns, c.pairing);
            const std::vector<double> samples = values(c.batch * plan.sample_scalars(), 6);
            const std::vector<double> spectrum = reference_forward(plan, samples, c.batch);
            const std::vector<double> expected = reference_inverse(plan, spectrum, c.batch);

            const std::vector<double> got =
                run_device(queue(), plan, spectrum, expected.size(), c.batch, false);

            EXPECT_EQ(got, expected);
        }

        TEST_P(Medium_SYCLFftCases, InverseRecoversTheSamplesOfTheForward)
        {
            const DeviceFftCase& c = GetParam();
            const Kernels::FftPlan<double> plan(c.rows, c.columns, c.pairing);
            const std::vector<double> samples = values(c.batch * plan.sample_scalars(), 6);
            const std::vector<double> spectrum = reference_forward(plan, samples, c.batch);

            const std::vector<double> got =
                run_device(queue(), plan, spectrum, samples.size(), c.batch, false);

            expect_within_rounding(got, samples, c);
        }

        INSTANTIATE_TEST_SUITE_P(, Medium_SYCLFftCases, ::testing::ValuesIn(cases()), device_fft_case_name);

        /** @brief Runs the kernels on the host CPU queue. */
        class Medium_SYCLFft : public HostDeviceTest
        {

        };

        TEST_F(Medium_SYCLFft, TwoRunsAgreeBitForBit)
        {
            const Kernels::FftPlan<double> plan(3, 12, Kernels::FftPairing::RealToComplex);
            const std::vector<double> input = values(2 * plan.sample_scalars(), 7);

            const std::vector<double> first =
                run_device(queue(), plan, input, 2 * plan.spectrum_scalars(), 2, true);
            const std::vector<double> second =
                run_device(queue(), plan, input, 2 * plan.spectrum_scalars(), 2, true);

            EXPECT_EQ(first, second);
        }
    } // namespace Test
} // namespace SushiBLAS
