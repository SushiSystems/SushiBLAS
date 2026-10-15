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
 * @brief Tests the partial sizes, the lease rule and an index reduction recorded on an engine.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiRuntime/core/operation_id.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/engine/execution_mode.hpp>
#include <SushiBLAS/reduction/reduce_1d.hpp>
#include <SushiBLAS/reductions/arg_reduction.hpp>
#include <SushiBLAS/reductions/dispatch.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/execution_mode_sweep/execution_mode_sweep.hpp>
#include <SushiBLAS/graph_probe/graph_probe.hpp>
#include <SushiBLAS/tensor_expectations/tensor_readback.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Records reductions on the fixture's engine. */
        class Unit_ReductionDispatch : public SushiBLASTest
        {

        };

        TEST_F(Unit_ReductionDispatch, AHalfPartialIsHeldWideAndTheRestAtTheirWidth)
        {
            EXPECT_EQ(Dispatch::partial_bytes(Core::DataType::Half),
                      sizeof(Reduce::AccumulationOf<sycl::half>));
            EXPECT_EQ(Dispatch::partial_bytes(Core::DataType::Float64), sizeof(double));
            EXPECT_EQ(Dispatch::partial_bytes(Core::DataType::Int32), sizeof(int32_t));
        }

        TEST_F(Unit_ReductionDispatch, AnEmptyTensorPlansNoPartialsAndTheReferenceLeasesNone)
        {
            GraphProbe reference(ExecutionMode::ForceReference);
            GraphProbe automatic;

            EXPECT_EQ(Dispatch::planned_partials_for(engine->create_tensor({0})), std::size_t{0});
            EXPECT_EQ(Dispatch::leased_partials(reference.recorder(), 12), std::size_t{0});
            EXPECT_EQ(Dispatch::leased_partials(automatic.recorder(), 12), std::size_t{12});
        }

        TEST_F(Unit_ReductionDispatch, AnIndexReductionGivesTheSameIndexInBothModes)
        {
            const Tensor t = create_initialized_tensor<float>({5}, {2, 7, -3, 7, -3});

            for (const ExecutionMode mode : K_BOTH_MODES)
            {
                Tensor result = create_initialized_tensor<int64_t>({1}, {-1});
                engine->recorder().set_mode(mode);
                Dispatch::execute_index_reduction(
                    engine->recorder(), t, result, "argmax", SushiRuntime::OperationID(),
                    Dispatch::ArgReductionDispatch<Reduce::RunningArgMax>{},
                    Dispatch::planned_partials_for(t));
                ASSERT_TRUE(engine->execute().has_value());

                EXPECT_EQ(read_tensor<int64_t>(result), (std::vector<int64_t>{1}));
            }
        }
    } // namespace Test
} // namespace SushiBLAS
