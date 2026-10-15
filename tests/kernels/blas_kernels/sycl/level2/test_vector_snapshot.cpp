/**************************************************************************/
/* test_vector_snapshot.cpp                                               */
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
 * @file test_vector_snapshot.cpp
 * @brief Tests that VectorSnapshot copies a strided vector into a contiguous one.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/blas_kernels/sycl/level2/vector_snapshot.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        TEST(Unit_VectorSnapshot, CopiesEveryIncxthElementOnTheHost)
        {
            const std::vector<float> x{1.0f, 9.0f, 2.0f, 9.0f, 3.0f};
            std::vector<float> snapshot(3, 0.0f);
            const Kernels::SYCL::Detail::VectorSnapshot<float> copy{x.data(), 2, snapshot.data()};

            for (std::size_t i = 0; i < snapshot.size(); ++i)
                copy(i);

            EXPECT_EQ(snapshot, (std::vector<float>{1.0f, 2.0f, 3.0f}));
        }

        /** @brief Runs the snapshot on the host CPU queue. */
        class Device_VectorSnapshot : public HostDeviceTest
        {

        };

        TEST_F(Device_VectorSnapshot, AKernelCopiesTheStridedElements)
        {
            std::vector<double> values(30);
            for (std::size_t i = 0; i < values.size(); ++i)
                values[i] = static_cast<double>(i);
            const SharedArray<double> x(queue(), values);
            SharedArray<double> snapshot(queue(), 10);
            const Kernels::SYCL::Detail::VectorSnapshot<double> copy{x.get(), 3, snapshot.get()};

            queue().parallel_for(sycl::range<1>(10), [=](sycl::id<1> i) { copy(i[0]); }).wait();

            for (std::size_t i = 0; i < 10; ++i)
                EXPECT_EQ(snapshot.get()[i], static_cast<double>(3 * i));
        }
    } // namespace Test
} // namespace SushiBLAS
