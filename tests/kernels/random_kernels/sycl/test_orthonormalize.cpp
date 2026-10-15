/**************************************************************************/
/* test_orthonormalize.cpp                                                */
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
 * @file test_orthonormalize.cpp
 * @brief Tests the work-group SYCL sweep against the reference sweep on the host device.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/orthonormal/orthonormal_sweep.hpp>
#include <SushiBLAS/random_kernels/reference/orthonormalize.hpp>
#include <SushiBLAS/random_kernels/sycl/orthonormalize.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Runs the sweep on the host CPU queue. */
        class Device_SYCLOrthonormalize : public HostDeviceTest
        {

        };

        TEST_F(Device_SYCLOrthonormalize, MatchesTheReferenceWithinItsSummationOrder)
        {
            const Kernels::VectorSet set{6, 50, 1, 50};
            std::vector<double> p = small_integers<double>(300, 1);
            for (int64_t v = 0; v < set.count; ++v)
                p[static_cast<std::size_t>(v * 50 + v)] += 9.0;
            SharedArray<double> device(queue(), p);

            Kernels::SYCL::Random::orthonormalize(queue(), {}, device.get(), set, 3.0).wait();
            Kernels::Reference::Random::orthonormalize(p.data(), set, 3.0);

            // The tree sum and the serial sum add in different orders.
            for (std::size_t i = 0; i < p.size(); ++i)
                EXPECT_NEAR(device.get()[i], p[i], 1e-12) << "element " << i;
        }

        TEST(Unit_SYCLOrthonormalize, TheGroupIsAPowerOfTwoTheDeviceAccepts)
        {
            sycl::queue* queue = host_queue();
            if (queue == nullptr)
                GTEST_SKIP() << "SYCL reports no CPU device.";

            const std::size_t group = Kernels::SYCL::Random::Detail::ortho_group_size(*queue);

            EXPECT_GE(group, std::size_t{1});
            EXPECT_LE(group, Kernels::SYCL::Random::Detail::K_ORTHO_MAX_GROUP);
            EXPECT_EQ(group & (group - 1), std::size_t{0});
        }
    } // namespace Test
} // namespace SushiBLAS
