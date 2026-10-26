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
 * @brief Tests that the SYCL batched FFT equals the reference bit for bit on the host device.
 * @author Mustafa Garip
 */

#include <SushiBLAS/signal_kernels/sycl/fft.hpp>

#include <limits>
#include <vector>
#include <cstddef>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/signal_kernels/fft_plan.hpp>
#include <SushiBLAS/signal_kernels/reference/fft.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>

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

            /** @brief Returns radix-2, Bluestein, packed and unpacked real, 1-D and 2-D cases. */
            std::vector<DeviceFftCase> cases()
            {
                using P = Kernels::FftPairing;
                return {{"c2c radix-2", 1, 32, 3, P::ComplexToComplex},
                        {"c2c Bluestein", 1, 10, 2, P::ComplexToComplex},
                        {"c2c 2-D", 4, 6, 2, P::ComplexToComplex},
                        {"r2c packed", 1, 16, 2, P::RealToComplex},
                        {"r2c packed Bluestein", 3, 12, 2, P::RealToComplex},
                        {"r2c odd", 5, 7, 1, P::RealToComplex}};
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
        } // namespace

        /** @brief Runs the kernels on the host CPU queue. */
        class Medium_SYCLFft : public HostDeviceTest
        {

        };

        TEST_F(Medium_SYCLFft, ForwardEqualsTheReferenceBitForBit)
        {
            for (const DeviceFftCase& c : cases())
            {
                SCOPED_TRACE(c.id);
                const Kernels::FftPlan<double> plan(c.rows, c.columns, c.pairing);
                const std::vector<double> input = values(c.batch * plan.sample_scalars(), 5);
                std::vector<double> expected(c.batch * plan.spectrum_scalars());
                Kernels::Reference::fft_forward(plan, input.data(), expected.data(), c.batch);

                const std::vector<double> got =
                    run_device(queue(), plan, input, expected.size(), c.batch, true);

                EXPECT_EQ(got, expected);
            }
        }

        TEST_F(Medium_SYCLFft, InverseEqualsTheReferenceBitForBit)
        {
            for (const DeviceFftCase& c : cases())
            {
                SCOPED_TRACE(c.id);
                const Kernels::FftPlan<double> plan(c.rows, c.columns, c.pairing);
                const std::vector<double> samples = values(c.batch * plan.sample_scalars(), 6);
                std::vector<double> spectrum(c.batch * plan.spectrum_scalars());
                Kernels::Reference::fft_forward(plan, samples.data(), spectrum.data(), c.batch);
                std::vector<double> expected(samples.size());
                Kernels::Reference::fft_inverse(plan, spectrum.data(), expected.data(), c.batch);

                const std::vector<double> got =
                    run_device(queue(), plan, spectrum, expected.size(), c.batch, false);

                const double bound = 64.0 * std::numeric_limits<double>::epsilon() *
                                     static_cast<double>(c.rows * c.columns);
                EXPECT_EQ(got, expected);
                for (std::size_t i = 0; i < expected.size(); ++i)
                    EXPECT_NEAR(got[i], samples[i], bound) << i;
            }
        }

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
