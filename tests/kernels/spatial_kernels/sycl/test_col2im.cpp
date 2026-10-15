/**************************************************************************/
/* test_col2im.cpp                                                        */
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
 * @file test_col2im.cpp
 * @brief Tests the SYCL col2im against the reference col2im on the host device.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/geometry/spatial_geometry.hpp>
#include <SushiBLAS/spatial_kernels/reference/col2im.hpp>
#include <SushiBLAS/spatial_kernels/sycl/col2im.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>
#include "../nhwc_span.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Runs the kernel on the host CPU queue. */
        class Device_SYCLCol2im : public HostDeviceTest
        {

        };

        TEST_F(Device_SYCLCol2im, MatchesTheReference)
        {
            // Every value is a small integer, so both kernels compute without rounding.
            const Spatial::WindowSpan span = nhwc_span(2, 9, 8, 3, uneven_geometry(), 1);
            const std::size_t inputs = static_cast<std::size_t>(span.input_elements());
            const Spatial::ColumnSpan cols{span.patch_elements(), 1};
            const std::vector<double> y = small_integers<double>(span.column_elements(), 1);
            const SharedArray<double> device_y(queue(), y);
            SharedArray<double> device_out(queue(), inputs);
            std::vector<double> out(inputs);

            Kernels::SYCL::col2im(queue(), span, cols, device_y.get(), device_out.get(),
                                  {}).wait();
            Kernels::Reference::col2im(span, cols, y.data(), out.data());

            EXPECT_EQ(device_out.values(), out);
        }
    } // namespace Test
} // namespace SushiBLAS
