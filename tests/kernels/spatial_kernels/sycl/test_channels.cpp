/**************************************************************************/
/* test_channels.cpp                                                      */
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
 * @file test_channels.cpp
 * @brief Tests the SYCL channel copy against the reference channel copy on the host device.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/geometry/spatial_geometry.hpp>
#include <SushiBLAS/spatial_kernels/reference/channels.hpp>
#include <SushiBLAS/spatial_kernels/sycl/channels.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>
#include "../nhwc_span.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Runs the kernel on the host CPU queue. */
        class Device_SYCLChannels : public HostDeviceTest
        {

        };

        TEST_F(Device_SYCLChannels, MatchesTheReference)
        {
            // Every value is a small integer, so both kernels compute without rounding.
            const Spatial::WindowSpan span = nhwc_span(2, 5, 4, 6, Spatial::ConvGeometry{}, 0);
            const std::size_t outputs = static_cast<std::size_t>(span.output_elements());
            const std::vector<double> x = small_integers<double>(span.input_elements(), 1);
            const SharedArray<double> device_x(queue(), x);
            SharedArray<double> device_out(queue(), outputs);
            std::vector<double> out(outputs, 0.0);

            Kernels::SYCL::copy_channel_range(queue(), span, 2, 1, 3, device_x.get(),
                                              device_out.get(), {}).wait();
            Kernels::Reference::copy_channel_range(span, 2, 1, 3, x.data(), out.data());

            EXPECT_EQ(device_out.values(), out);
        }
    } // namespace Test
} // namespace SushiBLAS
