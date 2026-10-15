/**************************************************************************/
/* test_dispatch.cpp                                                      */
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
 * @file test_dispatch.cpp
 * @brief Tests the dtypes a random task admits, its seed slots and the fill on both paths.
 * @author Mustafa Garip
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/engine/dispatch.hpp>
#include <SushiBLAS/engine/execution_mode.hpp>
#include <SushiBLAS/engine/kernel_pair_dispatcher.hpp>
#include <SushiBLAS/engine/task_description.hpp>
#include <SushiBLAS/random/dispatch.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/host_device/host_device_test.hpp>
#include <SushiBLAS/host_device/shared_array.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        using Dispatch::Domain;
        using Dispatch::Sampling;

        /// @brief An odd element count, so the last block of two is cut short.
        constexpr std::size_t K_COUNT = 1001;

        /** @brief Draws block b as the two positions it covers, so element i holds i. */
        struct PositionBlock
        {
            /** @brief Returns the two values of block b. */
            std::array<float, 2> operator()(uint64_t b) const
            {
                return {static_cast<float>(2 * b), static_cast<float>(2 * b + 1)};
            }
        };

        /** @brief Builds tensors for the dtype check on the fixture's engine. */
        class Medium_RandomDispatch : public SushiBLASTest
        {

        };

        /** @brief Runs the fill on the host CPU queue. */
        class Medium_RandomFill : public HostDeviceTest
        {

        };

        TEST_F(Medium_RandomDispatch, DrawingRefusesHalfAndEveryDomainRefusesItsOutsiders)
        {
            EXPECT_FALSE(Dispatch::random_admits(Domain::Real, Sampling::DrawsValues,
                                                 Core::DataType::Half));
            EXPECT_TRUE(Dispatch::random_admits(Domain::Real, Sampling::MovesValues,
                                                Core::DataType::Half));
            EXPECT_FALSE(Dispatch::random_admits(Domain::RealAndComplex, Sampling::DrawsValues,
                                                 Core::DataType::Int32));
            EXPECT_ANY_THROW(Dispatch::require_random_dtype(
                engine->create_tensor({2}, Core::DataType::Int64), Domain::Real,
                Sampling::DrawsValues, "draw"));
        }

        TEST_F(Medium_RandomDispatch, TheSeedAndTheOffsetTakeTheLastTwoSlots)
        {
            TaskDescription description;

            Dispatch::set_rng_params(description, 42, 7);

            EXPECT_EQ(description.parameter<uint64_t>(Dispatch::K_RNG_SEED_PARAM), uint64_t{42});
            EXPECT_EQ(description.parameter<uint64_t>(Dispatch::K_RNG_OFFSET_PARAM), uint64_t{7});
        }

        TEST_F(Medium_RandomFill, BothPathsPlaceEveryBlockAtItsPositions)
        {
            SharedArray<float> by_reference(queue(), K_COUNT);
            SharedArray<float> by_sycl(queue(), K_COUNT);
            const KernelPairDispatcher reference(ExecutionMode::ForceReference);
            const KernelPairDispatcher sycl_path(ExecutionMode::ForceSYCL);
            const std::vector<sycl::event> none;

            Dispatch::RandomFill(reference, queue(), none)(static_cast<int64_t>(K_COUNT),
                                                           by_reference.get(), PositionBlock{})
                .wait();
            Dispatch::RandomFill(sycl_path, queue(), none)(static_cast<int64_t>(K_COUNT),
                                                           by_sycl.get(), PositionBlock{})
                .wait();

            EXPECT_EQ(by_sycl.values(), by_reference.values());
            EXPECT_EQ(by_reference.get()[K_COUNT - 1], static_cast<float>(K_COUNT - 1));
        }
    } // namespace Test
} // namespace SushiBLAS
