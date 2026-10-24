/**************************************************************************/
/* test_convolve_1d.cpp                                                   */
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
 * @file test_convolve_1d.cpp
 * @brief Tests the SYCL valid-mode convolution against the reference one.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/signal_kernels/reference/convolve_1d.hpp>
#include <SushiBLAS/signal_kernels/sycl/convolve_1d.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Runs the kernel on the host CPU queue. */
        class Medium_SYCLConvolve1D : public HostDeviceTest
        {

        };

        TEST_F(Medium_SYCLConvolve1D, MatchesTheReferenceOnBatchedRows)
        {
            // Every value is a small integer, so both kernels compute without rounding.
            constexpr int64_t K_ROWS = 6;
            constexpr int64_t K_LENGTH = 37;
            constexpr int64_t K_TAPS = 9;
            constexpr std::size_t K_OUTPUTS = K_ROWS * (K_LENGTH - K_TAPS + 1);
            const std::vector<double> signal = small_integers<double>(K_ROWS * K_LENGTH, 1);
            const std::vector<double> taps = small_integers<double>(K_TAPS, 2);
            const SharedArray<double> device_signal(queue(), signal);
            const SharedArray<double> device_taps(queue(), taps);
            SharedArray<double> device_out(queue(), K_OUTPUTS);
            std::vector<double> out(K_OUTPUTS);

            Kernels::SYCL::convolve_1d(queue(), K_ROWS, K_LENGTH, K_TAPS, device_signal.get(),
                                       device_taps.get(), device_out.get(), {}).wait();
            Kernels::Reference::convolve_1d(K_ROWS, K_LENGTH, K_TAPS, signal.data(), taps.data(),
                                            out.data());

            EXPECT_EQ(device_out.values(), out);
        }

        TEST_F(Medium_SYCLConvolve1D, MatchesTheReferenceWithOneTapAndWithTapsAsLongAsTheRow)
        {
            constexpr int64_t K_ROWS = 4;
            constexpr int64_t K_LENGTH = 7;

            for (const int64_t tap_count : {int64_t{1}, K_LENGTH})
            {
                const std::size_t outputs =
                    static_cast<std::size_t>(K_ROWS * (K_LENGTH - tap_count + 1));
                const std::vector<double> signal = small_integers<double>(K_ROWS * K_LENGTH, 3);
                const std::vector<double> taps =
                    small_integers<double>(static_cast<std::size_t>(tap_count), 4);
                const SharedArray<double> device_signal(queue(), signal);
                const SharedArray<double> device_taps(queue(), taps);
                SharedArray<double> device_out(queue(), outputs);
                std::vector<double> out(outputs);

                Kernels::SYCL::convolve_1d(queue(), K_ROWS, K_LENGTH, tap_count,
                                           device_signal.get(), device_taps.get(),
                                           device_out.get(), {}).wait();
                Kernels::Reference::convolve_1d(K_ROWS, K_LENGTH, tap_count, signal.data(),
                                                taps.data(), out.data());

                EXPECT_EQ(device_out.values(), out) << "taps " << tap_count;
            }
        }
    } // namespace Test
} // namespace SushiBLAS
