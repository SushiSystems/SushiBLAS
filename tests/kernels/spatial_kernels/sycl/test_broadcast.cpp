/**************************************************************************/
/* test_broadcast.cpp                                                     */
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
 * @file test_broadcast.cpp
 * @brief Tests the SYCL affine, broadcast and layout conversion against the reference.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/geometry/spatial_geometry.hpp>
#include <SushiBLAS/spatial_kernels/reference/broadcast.hpp>
#include <SushiBLAS/spatial_kernels/sycl/broadcast.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>
#include "../nhwc_span.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Runs the kernel on the host CPU queue. */
        class Device_SYCLBroadcast : public HostDeviceTest
        {

        };

        TEST_F(Device_SYCLBroadcast, MatchesTheReference)
        {
            // Every value is a small integer, so both kernels compute without rounding.
            const Spatial::WindowSpan span = nhwc_span(2, 5, 4, 3, Spatial::ConvGeometry{}, 0);
            const std::size_t outputs = static_cast<std::size_t>(span.output_elements());
            const std::vector<double> x = small_integers<double>(span.input_elements(), 1);
            const std::vector<double> gamma = small_integers<double>(3, 2);
            const std::vector<double> beta = small_integers<double>(3, 3);
            const std::vector<double> per_channel = small_integers<double>(6, 4);
            const SharedArray<double> device_x(queue(), x);
            const SharedArray<double> device_gamma(queue(), gamma);
            const SharedArray<double> device_beta(queue(), beta);
            const SharedArray<double> device_per_channel(queue(), per_channel);
            SharedArray<double> affine(queue(), outputs);
            SharedArray<double> broadcast(queue(), outputs);
            SharedArray<double> converted(queue(), outputs);
            std::vector<double> expected_affine(outputs);
            std::vector<double> expected_broadcast(outputs);
            std::vector<double> expected_converted(outputs);

            Kernels::SYCL::channel_affine(queue(), span, device_x.get(), device_gamma.get(),
                                          device_beta.get(), 1, 1, affine.get(), {}).wait();
            Kernels::SYCL::broadcast_spatial(queue(), span, device_per_channel.get(), 3, 1,
                                             broadcast.get(), 2.0, {}).wait();
            Kernels::SYCL::layout_convert(queue(), span, device_x.get(), converted.get(),
                                          {}).wait();
            Kernels::Reference::channel_affine(span, x.data(), gamma.data(), beta.data(), 1, 1,
                                               expected_affine.data());
            Kernels::Reference::broadcast_spatial(span, per_channel.data(), 3, 1,
                                                  expected_broadcast.data(), 2.0);
            Kernels::Reference::layout_convert(span, x.data(), expected_converted.data());

            EXPECT_EQ(affine.values(), expected_affine);
            EXPECT_EQ(broadcast.values(), expected_broadcast);
            EXPECT_EQ(converted.values(), expected_converted);
        }
    } // namespace Test
} // namespace SushiBLAS
