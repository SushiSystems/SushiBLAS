/**************************************************************************/
/* test_linalg.cpp                                                        */
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
 * @file test_linalg.cpp
 * @brief Tests that linalg.hpp alone reaches the reference and the SYCL linear algebra kernels.
 * @author Mustafa Garip
 */

#include <SushiBLAS/linalg_kernels/linalg.hpp>

#include <vector>
#include <cstdint>
#include <gtest/gtest.h>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Small_LinalgKernels, AReferenceKernelIsReachable)
        {
            std::vector<double> a{4.0};
            std::vector<std::uint32_t> pivots(1);
            std::vector<std::uint32_t> singular(1, 7u);
            std::vector<double> x{8.0};

            Kernels::Reference::lu_factor(a.data(), pivots.data(), singular.data(), 1, 1);
            Kernels::Reference::lu_solve(a.data(), pivots.data(), x.data(), 1, 1, 1);

            EXPECT_EQ(singular[0], 0u);
            EXPECT_EQ(x[0], 2.0);
        }

        /** @brief Runs a kernel reached through linalg.hpp on the host CPU queue. */
        class Medium_LinalgKernelsOnDevice : public HostDeviceTest
        {

        };

        TEST_F(Medium_LinalgKernelsOnDevice, ASYCLKernelIsReachable)
        {
            SharedArray<double> a(queue(), std::vector<double>{4.0});
            SharedArray<std::uint32_t> pivots(queue(), 1);
            SharedArray<std::uint32_t> singular(queue(), 1);

            Kernels::SYCL::lu_factor(queue(), a.get(), pivots.get(), singular.get(), 1, 1, {})
                .wait();

            EXPECT_EQ(singular.values()[0], 0u);
            EXPECT_EQ(a.values()[0], 4.0);
        }
    } // namespace Test
} // namespace SushiBLAS
