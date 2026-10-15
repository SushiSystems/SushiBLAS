/**************************************************************************/
/* test_rotm.cpp                                                          */
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
 * @file test_rotm.cpp
 * @brief Tests the SYCL rotm against the reference rotm under every flag on the host device.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/blas_kernels/reference/level1/rotm.hpp>
#include <SushiBLAS/blas_kernels/sycl/level1/rotm.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief The vector length.
        constexpr int64_t K_LENGTH = 1000;

        /** @brief Runs rotm on the host CPU queue. */
        class Medium_SYCLRotm : public HostDeviceTest
        {

        };

        TEST_F(Medium_SYCLRotm, MatchesTheReferenceUnderEveryFlag)
        {
            for (const float flag : {-2.0f, -1.0f, 0.0f, 1.0f})
            {
                SCOPED_TRACE(flag);
                const std::vector<float> param{flag, 2.0f, -1.0f, 3.0f, -2.0f};
                std::vector<float> x = small_integers<float>(K_LENGTH, 1);
                std::vector<float> y = small_integers<float>(K_LENGTH, 2);
                SharedArray<float> device_x(queue(), x);
                SharedArray<float> device_y(queue(), y);
                const SharedArray<float> device_param(queue(), param);

                Kernels::SYCL::rotm<float>(queue(), K_LENGTH / 2, device_x.get(), 2,
                                           device_y.get(), 2, device_param.get(), {}).wait();
                Kernels::Reference::rotm<float>(K_LENGTH / 2, x.data(), 2, y.data(), 2,
                                                param.data());

                EXPECT_EQ(device_x.values(), x);
                EXPECT_EQ(device_y.values(), y);
            }
        }
    } // namespace Test
} // namespace SushiBLAS
