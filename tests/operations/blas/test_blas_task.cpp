/**************************************************************************/
/* test_blas_task.cpp                                                     */
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
 * @file test_blas_task.cpp
 * @brief Tests that a BLAS task records for a real or complex dtype and refuses an integer one.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/engine/kernel_pair_dispatcher.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/task_description.hpp>
#include <SushiBLAS/blas/blas_task.hpp>
#include <SushiBLAS/graph_probe/graph_probe.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief A BLAS body that runs nothing on either path. */
        struct NothingBody
        {
            /** @brief Returns a completed event for any element type. */
            template<typename T>
            sycl::event operator()(T, const KernelPairDispatcher& dispatcher, sycl::queue& q,
                                   const std::vector<sycl::event>& deps) const
            {
                return dispatcher.run(q, deps,
                                      []()
                                      {

                                      },
                                      [](sycl::queue&, const std::vector<sycl::event>& events)
                                      {
                                          sycl::event::wait(events);
                                          return sycl::event();
                                      });
            }
        };

        TEST(Medium_BLASTask, RecordsOneTaskForARealOrAComplexDtype)
        {
            GraphProbe probe;
            TaskDescription description;
            description.name = "nothing";

            Internal::execute_blas_task(probe.recorder(), description, OperandSet(),
                                        Core::DataType::Float32, NothingBody{});
            Internal::execute_blas_task(probe.recorder(), description, OperandSet(),
                                        Core::DataType::Complex64, NothingBody{});

            EXPECT_EQ(probe.recorded_count(), std::size_t{2});
        }

        TEST(Medium_BLASTask, RefusesAnIntegerDtypeAndRecordsNothing)
        {
            GraphProbe probe;
            TaskDescription description;
            description.name = "nothing";

            EXPECT_ANY_THROW(Internal::execute_blas_task(probe.recorder(), description,
                                                         OperandSet(), Core::DataType::Int64,
                                                         NothingBody{}));
            EXPECT_EQ(probe.recorded_count(), std::size_t{0});
        }
    } // namespace Test
} // namespace SushiBLAS
