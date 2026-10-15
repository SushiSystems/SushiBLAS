/**************************************************************************/
/* test_rotg.cpp                                                          */
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
 * @file test_rotg.cpp
 * @brief Tests the SYCL rotg against the reference rotg on the host device.
 * @author Mustafa Garip
 */

#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/blas_kernels/reference/level1/rotg.hpp>
#include <SushiBLAS/blas_kernels/sycl/level1/rotg.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Runs rotg on the host CPU queue. */
        class Medium_SYCLRotg : public HostDeviceTest
        {

        };

        TEST_F(Medium_SYCLRotg, MatchesTheReferenceBitForBit)
        {
            const std::vector<std::vector<double>> pairs{
                {3.0, 4.0}, {-4.0, 3.0}, {0.0, 2.0}, {5.0, 0.0}, {1e-200, 3e-200}};

            for (const std::vector<double>& pair : pairs)
            {
                SCOPED_TRACE(pair[0]);
                SharedArray<double> values(queue(), std::vector<double>{pair[0], pair[1], 0, 0});
                double* v = values.get();
                Kernels::SYCL::rotg(queue(), v, v + 1, v + 2, v + 3, {}).wait();

                double a = pair[0];
                double b = pair[1];
                double c = 0.0;
                double s = 0.0;
                Kernels::Reference::rotg(&a, &b, &c, &s);

                EXPECT_EQ(values.values(), (std::vector<double>{a, b, c, s}));
            }
        }
    } // namespace Test
} // namespace SushiBLAS
