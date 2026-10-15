/**************************************************************************/
/* test_task.cpp                                                          */
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
 * @file test_task.cpp
 * @brief Tests that a typed task records for an admitted dtype, runs its arm and refuses the rest.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/engine/dispatch.hpp>
#include <SushiBLAS/engine/kernel_pair_dispatcher.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/task.hpp>
#include <SushiBLAS/engine/task_description.hpp>
#include <SushiBLAS/graph_probe/graph_probe.hpp>
#include <SushiBLAS/test_runner/test_environment.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Records the size of the element type the body ran for. */
        struct SizeBody
        {
            int* bytes; /**< Receives sizeof the element type, on the host. */

            /** @brief Runs on the host and records sizeof T. */
            template<typename T>
            sycl::event operator()(T, const KernelPairDispatcher& dispatcher, sycl::queue& q,
                                   const std::vector<sycl::event>& deps) const
            {
                int* out = bytes;
                return dispatcher.run(q, deps, [out]() { *out = static_cast<int>(sizeof(T)); },
                                      [](sycl::queue&, const std::vector<sycl::event>&)
                                      {
                                          return sycl::event();
                                      });
            }
        };

        TEST(Unit_Task, RecordsOneTaskThatRunsTheArmOfItsDtype)
        {
            GraphProbe probe(ExecutionMode::ForceReference);
            TaskDescription description;
            description.name = "size";
            int bytes = 0;

            Internal::execute_typed_task(probe.recorder(), description, OperandSet(),
                                         Core::DataType::Float64, "test", SizeBody{&bytes});
            EXPECT_EQ(probe.recorded_count(), std::size_t{1});
            ASSERT_TRUE(probe.execute().has_value());
            ASSERT_TRUE(TestEnvironment::runtime().wait().has_value());

            EXPECT_EQ(bytes, static_cast<int>(sizeof(double)));
        }

        TEST(Unit_Task, RefusesADtypeItsDomainDoesNotAdmit)
        {
            GraphProbe probe;
            TaskDescription description;
            description.name = "size";
            int bytes = 0;

            EXPECT_ANY_THROW(Internal::execute_typed_task<Dispatch::Domain::Real>(
                probe.recorder(), description, OperandSet(), Core::DataType::Complex32, "test",
                SizeBody{&bytes}));
            EXPECT_EQ(probe.recorded_count(), std::size_t{0});
        }
    } // namespace Test
} // namespace SushiBLAS
