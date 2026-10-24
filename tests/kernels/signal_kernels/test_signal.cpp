/**************************************************************************/
/* test_signal.cpp                                                        */
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
 * @file test_signal.cpp
 * @brief Tests that signal.hpp alone reaches the reference and the SYCL signal kernels.
 * @author Mustafa Garip
 */

#include <SushiBLAS/signal_kernels/signal.hpp>

#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Small_SignalKernels, AReferenceKernelIsReachable)
        {
            const std::vector<float> signal{1.0f, 2.0f, 3.0f};
            const std::vector<float> taps{1.0f, 1.0f};
            std::vector<float> out(2, 0.0f);

            Kernels::Reference::convolve_1d(1, 3, 2, signal.data(), taps.data(), out.data());

            EXPECT_EQ(out, (std::vector<float>{3.0f, 5.0f}));
        }

        /** @brief Runs a kernel reached through signal.hpp on the host CPU queue. */
        class Medium_SignalKernelsOnDevice : public HostDeviceTest
        {

        };

        TEST_F(Medium_SignalKernelsOnDevice, ASYCLKernelIsReachable)
        {
            const SharedArray<float> signal(queue(), std::vector<float>{1.0f, 2.0f, 3.0f});
            const SharedArray<float> taps(queue(), std::vector<float>{1.0f, 1.0f});
            SharedArray<float> out(queue(), 2);

            Kernels::SYCL::convolve_1d(queue(), 1, 3, 2, signal.get(), taps.get(), out.get(), {})
                .wait();

            EXPECT_EQ(out.values(), (std::vector<float>{3.0f, 5.0f}));
        }
    } // namespace Test
} // namespace SushiBLAS
