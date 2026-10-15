/**************************************************************************/
/* test_kernel_pair_dispatcher.cpp                                        */
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
 * @file test_kernel_pair_dispatcher.cpp
 * @brief Tests that the dispatcher picks the reference body under ForceReference only.
 * @author Mustafa Garip
 */

#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/engine/execution_mode.hpp>
#include <SushiBLAS/engine/kernel_pair_dispatcher.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Returns which body a dispatcher in mode ran: 1 for reference, 2 for SYCL. */
        static int body_run_in(ExecutionMode mode)
        {
            int ran = 0;
            const KernelPairDispatcher dispatcher(mode);

            dispatcher.run(TestEnvironment::queue(), {}, [&ran]() { ran = 1; },
                           [&ran](sycl::queue&, const std::vector<sycl::event>&)
                           {
                               ran = 2;
                               return sycl::event();
                           }).wait();
            return ran;
        }

        TEST(Unit_KernelPairDispatcher, ForceReferenceRunsTheReferenceBody)
        {
            EXPECT_EQ(body_run_in(ExecutionMode::ForceReference), 1);
        }

        TEST(Unit_KernelPairDispatcher, EveryOtherModeRunsTheSYCLBody)
        {
            EXPECT_EQ(body_run_in(ExecutionMode::Auto), 2);
            EXPECT_EQ(body_run_in(ExecutionMode::ForceSYCL), 2);
        }

        TEST(Unit_KernelPairDispatcher, TheReferenceBodyRunsAfterItsDependencies)
        {
            sycl::queue& queue = TestEnvironment::queue();
            int* flag = sycl::malloc_shared<int>(1, queue);
            *flag = 0;
            const sycl::event writer = queue.single_task([=]() { *flag = 7; });
            int seen = -1;

            KernelPairDispatcher(ExecutionMode::ForceReference)
                .run(queue, {writer}, [&]() { seen = *flag; },
                     [](sycl::queue&, const std::vector<sycl::event>&) { return sycl::event(); })
                .wait();

            EXPECT_EQ(seen, 7);
            sycl::free(flag, queue);
        }
    } // namespace Test
} // namespace SushiBLAS
