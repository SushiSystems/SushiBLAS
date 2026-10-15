/**************************************************************************/
/* test_blas.cpp                                                          */
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
 * @file test_blas.cpp
 * @brief Tests that blas.hpp alone reaches a reference and a SYCL kernel of each BLAS level.
 * @author Mustafa Garip
 */

#include <SushiBLAS/blas_kernels/blas.hpp>

#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Unit_BlasKernels, TheReferenceKernelsOfEachLevelAreReachable)
        {
            std::vector<double> y{1.0, 1.0};
            const std::vector<double> x{2.0, 3.0};
            Kernels::Reference::axpy<double>(2, 1.0, x.data(), 1, y.data(), 1);

            const std::vector<double> a{1.0, 0.0, 0.0, 1.0};
            std::vector<double> out{0.0, 0.0};
            Kernels::Reference::gemv(Core::Layout::RowMajor, Core::Transpose::NoTrans, 2, 2, 1.0,
                                     a.data(), 2, y.data(), 1, 0.0, out.data(), 1);

            EXPECT_EQ(out, (std::vector<double>{3.0, 4.0}));
        }

        /** @brief Runs a kernel reached through blas.hpp on the host CPU queue. */
        class Device_BlasKernels : public HostDeviceTest
        {

        };

        TEST_F(Device_BlasKernels, ASYCLKernelIsReachable)
        {
            const SharedArray<float> x(queue(), std::vector<float>{1.0f, 2.0f});
            SharedArray<float> y(queue(), std::vector<float>{10.0f, 20.0f});

            Kernels::SYCL::axpy<float>(queue(), 2, 2.0f, x.get(), 1, y.get(), 1, {}).wait();

            EXPECT_EQ(y.values(), (std::vector<float>{12.0f, 24.0f}));
        }
    } // namespace Test
} // namespace SushiBLAS
