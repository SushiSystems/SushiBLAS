/**************************************************************************/
/* test_im2col.cpp                                                        */
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
 * @file test_im2col.cpp
 * @brief Tests the SYCL im2col against the reference im2col on the host device.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/geometry/spatial_geometry.hpp>
#include <SushiBLAS/spatial_kernels/reference/im2col.hpp>
#include <SushiBLAS/spatial_kernels/sycl/im2col.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>
#include "../nhwc_span.hpp"

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Runs the kernel on the host CPU queue. */
        class Medium_SYCLIm2col : public HostDeviceTest
        {

        };

        TEST_F(Medium_SYCLIm2col, MatchesTheReference)
        {
            // Every value is a small integer, so both kernels compute without rounding.
            const Spatial::WindowSpan span = nhwc_span(2, 9, 8, 3, uneven_geometry(), 1);
            const std::size_t columns = static_cast<std::size_t>(span.column_elements());
            const Spatial::ColumnSpan cols{span.patch_elements(), 1};
            const std::vector<double> x = small_integers<double>(span.input_elements(), 1);
            const SharedArray<double> device_x(queue(), x);
            SharedArray<double> device_out(queue(), columns);
            std::vector<double> out(columns);

            Kernels::SYCL::im2col(queue(), span, cols, device_x.get(), device_out.get(),
                                  {}).wait();
            Kernels::Reference::im2col(span, cols, x.data(), out.data());

            EXPECT_EQ(device_out.values(), out);
        }
    } // namespace Test
} // namespace SushiBLAS
