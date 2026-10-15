/**************************************************************************/
/* test_window_dot_weights.cpp                                            */
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
 * @file test_window_dot_weights.cpp
 * @brief Tests the SYCL weight gradient in both launch shapes against the reference.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/geometry/spatial_geometry.hpp>
#include <SushiBLAS/reduction/reduce_axis.hpp>
#include <SushiBLAS/spatial_kernels/reference/window_dot_weights.hpp>
#include <SushiBLAS/spatial_kernels/sycl/window_dot_weights.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>
#include "../nhwc_span.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Runs the kernel on the host CPU queue. */
        class Device_SYCLWindowDotWeights : public HostDeviceTest
        {

        };

        TEST_F(Device_SYCLWindowDotWeights, MatchesTheReference)
        {
            // Every value is a small integer, so both kernels compute without rounding.
            const Spatial::WindowSpan span = nhwc_span(2, 9, 8, 3, uneven_geometry(), 1);
            const Spatial::TapSpan taps{span.geometry.kernel_w * 3, 3, 1};
            const std::vector<double> x = small_integers<double>(span.input_elements(), 1);
            const std::vector<double> dy = small_integers<double>(span.output_elements(), 2);
            const SharedArray<double> device_x(queue(), x);
            const SharedArray<double> device_dy(queue(), dy);
            std::vector<double> dw(static_cast<std::size_t>(span.patch_elements()));
            Kernels::Reference::window_dot_weights_backward(span, taps, dy.data(), x.data(),
                                                            dw.data());
            for (const std::size_t local : {std::size_t{1}, std::size_t{16}})
            {
                SCOPED_TRACE(local);
                SharedArray<double> device_dw(queue(),
                                              static_cast<std::size_t>(span.patch_elements()));

                Kernels::SYCL::window_dot_weights_backward(queue(), span, taps,
                                                           Reduce::AxisPlan{local},
                                                           device_dy.get(), device_x.get(),
                                                           device_dw.get(), {}).wait();

                EXPECT_EQ(device_dw.values(), dw);
            }
        }
    } // namespace Test
} // namespace SushiBLAS
