/**************************************************************************/
/* test_map_task.cpp                                                      */
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
 * @file test_map_task.cpp
 * @brief Tests that a map task and a zip task give one result on the reference and the SYCL path.
 * @author Mustafa Garip
 */

#include <array>
#include <cstddef>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/engine/dispatch.hpp>
#include <SushiBLAS/engine/execution_mode.hpp>
#include <SushiBLAS/engine/kernel_pair_dispatcher.hpp>
#include <SushiBLAS/functor_map/map_task.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>
#include <SushiBLAS/small_integers/small_integers.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /// @brief The element count; not a multiple of any vector width.
        constexpr std::size_t K_COUNT = 1027;

        /** @brief Returns a * b - a, applied element by element. */
        struct MultiplyLess
        {
            /** @brief Applies the operation to one pair of elements. */
            template<typename T>
            T operator()(T a, T b) const
            {
                return a * b - a;
            }
        };

        /** @brief Swaps two elements and negates the first. */
        struct SwapNegate
        {
            /** @brief Applies the update to one pair of elements in place. */
            template<typename T>
            void operator()(T& a, T& b) const
            {
                const T held = a;
                a = -b;
                b = held;
            }
        };

        /** @brief Runs the tasks on the host CPU queue. */
        class Device_MapTask : public HostDeviceTest
        {

        };

        TEST_F(Device_MapTask, BothPathsOfAMapTaskWriteTheSameElements)
        {
            const SharedArray<float> a(queue(), small_integers<float>(K_COUNT, 1));
            const SharedArray<float> b(queue(), small_integers<float>(K_COUNT, 2));
            SharedArray<float> by_reference(queue(), K_COUNT);
            SharedArray<float> by_sycl(queue(), K_COUNT);

            Dispatch::Detail::make_map_task<Dispatch::Domain::All>(
                Core::DataType::Float32, "test", "map", K_COUNT, by_reference.get(),
                MultiplyLess{}, a.get(), b.get())(
                KernelPairDispatcher(ExecutionMode::ForceReference), queue(), {}).wait();
            Dispatch::Detail::make_map_task<Dispatch::Domain::All>(
                Core::DataType::Float32, "test", "map", K_COUNT, by_sycl.get(), MultiplyLess{},
                a.get(), b.get())(KernelPairDispatcher(ExecutionMode::ForceSYCL), queue(),
                                  {}).wait();

            EXPECT_EQ(by_sycl.values(), by_reference.values());
            EXPECT_EQ(by_reference.get()[0], a.get()[0] * b.get()[0] - a.get()[0]);
        }

        TEST_F(Device_MapTask, BothPathsOfAZipTaskLeaveTheSameElements)
        {
            const std::vector<double> first = small_integers<double>(K_COUNT, 3);
            const std::vector<double> second = small_integers<double>(K_COUNT, 4);
            SharedArray<double> ref_a(queue(), first);
            SharedArray<double> ref_b(queue(), second);
            SharedArray<double> sycl_a(queue(), first);
            SharedArray<double> sycl_b(queue(), second);

            Dispatch::Detail::ZipTask<Dispatch::Domain::All, SwapNegate, 2>(
                Core::DataType::Float64, "test", "zip", K_COUNT,
                std::array<void*, 2>{ref_a.get(), ref_b.get()}, SwapNegate{})(
                KernelPairDispatcher(ExecutionMode::ForceReference), queue(), {}).wait();
            Dispatch::Detail::ZipTask<Dispatch::Domain::All, SwapNegate, 2>(
                Core::DataType::Float64, "test", "zip", K_COUNT,
                std::array<void*, 2>{sycl_a.get(), sycl_b.get()}, SwapNegate{})(
                KernelPairDispatcher(ExecutionMode::ForceSYCL), queue(), {}).wait();

            EXPECT_EQ(sycl_a.values(), ref_a.values());
            EXPECT_EQ(sycl_b.values(), ref_b.values());
            EXPECT_EQ(ref_b.values(), first);
        }
    } // namespace Test
} // namespace SushiBLAS
