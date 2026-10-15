/**************************************************************************/
/* test_pool_max.cpp                                                      */
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
 * @file test_pool_max.cpp
 * @brief Tests the SYCL max pool and its backward against the reference ones.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/geometry/spatial_geometry.hpp>
#include <SushiBLAS/spatial_kernels/reference/pool_max.hpp>
#include <SushiBLAS/spatial_kernels/sycl/pool_max.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>
#include "../nhwc_span.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Runs the kernel on the host CPU queue. */
        class Device_SYCLPoolMax : public HostDeviceTest
        {

        };

        TEST_F(Device_SYCLPoolMax, MatchesTheReference)
        {
            // Every value is a small integer, so both kernels compute without rounding.
            const Spatial::WindowSpan span = nhwc_span(2, 9, 8, 3, uneven_geometry(), 1);
            const std::size_t outputs = static_cast<std::size_t>(span.output_elements());
            const std::size_t inputs = static_cast<std::size_t>(span.input_elements());
            const std::vector<double> x = small_integers<double>(span.input_elements(), 1);
            const std::vector<double> dy = small_integers<double>(span.output_elements(), 2);
            const SharedArray<double> device_x(queue(), x);
            const SharedArray<double> device_dy(queue(), dy);
            SharedArray<double> device_y(queue(), outputs);
            SharedArray<int64_t> device_index(queue(), outputs);
            SharedArray<double> device_dx(queue(), inputs);
            std::vector<double> y(outputs);
            std::vector<int64_t> index(outputs);
            std::vector<double> dx(inputs);

            Kernels::SYCL::pool_max(queue(), span, device_x.get(), device_y.get(),
                                    device_index.get(), {}).wait();
            Kernels::SYCL::pool_max_backward(queue(), span, device_dy.get(), device_index.get(),
                                             device_dx.get(), {}).wait();
            Kernels::Reference::pool_max(span, x.data(), y.data(), index.data());
            Kernels::Reference::pool_max_backward(span, dy.data(), index.data(), dx.data());

            EXPECT_EQ(device_y.values(), y);
            EXPECT_EQ(device_index.values(), index);
            EXPECT_EQ(device_dx.values(), dx);
        }
    } // namespace Test
} // namespace SushiBLAS
