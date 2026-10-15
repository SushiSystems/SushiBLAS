/**************************************************************************/
/* host_device_test.hpp                                                   */
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
 * @file host_device_test.hpp
 * @brief Declares HostDeviceTest, the fixture that skips a test on a machine with no CPU device.
 * @author Mustafa Garip
 */

#pragma once

#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/host_device/host_queue.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Gives a kernel test the host CPU queue, and skips the test when there is none. */
        class HostDeviceTest : public ::testing::Test
        {
            protected:
                /** @brief Skips the test when SYCL reports no CPU device. */
                void SetUp() override
                {
                    if (host_queue() == nullptr)
                        GTEST_SKIP() << "SYCL reports no CPU device.";
                }

                /** @brief Returns the host CPU queue. */
                [[nodiscard]] static sycl::queue& queue()
                {
                    return *host_queue();
                }
        };
    } // namespace Test
} // namespace SushiBLAS
