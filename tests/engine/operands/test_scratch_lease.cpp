/**************************************************************************/
/* test_scratch_lease.cpp                                                 */
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
 * @file test_scratch_lease.cpp
 * @brief Tests when a BLAS call leases scratch, how much, and that the lease is declared written.
 * @author Mustafa Garip
 */

#include <complex>
#include <cstddef>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/engine/execution_mode.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/substitution/running_solution.hpp>
#include <SushiBLAS/operands/scratch_lease.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/graph_probe/graph_probe.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Leases scratch through probes over tensors of the fixture's engine. */
        class Unit_ScratchLease : public SushiBLASTest
        {

        };

        TEST_F(Unit_ScratchLease, AnIntegerSolutionNeedsNoWorkspace)
        {
            EXPECT_EQ(Internal::running_solution_bytes(Core::DataType::Int32), std::size_t{0});
            EXPECT_EQ(Internal::running_solution_bytes(Core::DataType::Half),
                      Kernels::RunningSolution<sycl::half>::K_NEEDS_WORKSPACE
                          ? sizeof(Kernels::RunningSolution<sycl::half>::Wide)
                          : std::size_t{0});
        }

        TEST_F(Unit_ScratchLease, ALeaseIsDeclaredAsAWrite)
        {
            GraphProbe probe;
            const Tensor anchor = engine->create_tensor({4});
            OperandSet operands;

            const ScratchBuffer scratch =
                Internal::allocate_blas_scratch(probe.recorder(), anchor, 128, operands);

            EXPECT_EQ(scratch.bytes(), std::size_t{128});
            EXPECT_EQ(operands.write_regions().size(), std::size_t{1});
        }

        TEST_F(Unit_ScratchLease, ForceReferenceAndAnEmptyRequestLeaseNothing)
        {
            GraphProbe reference(ExecutionMode::ForceReference);
            GraphProbe automatic;
            const Tensor anchor = engine->create_tensor({4});
            OperandSet operands;

            EXPECT_TRUE(Internal::allocate_blas_scratch(reference.recorder(), anchor, 128,
                                                        operands).is_empty());
            EXPECT_TRUE(Internal::allocate_blas_scratch(automatic.recorder(), anchor, 0,
                                                        operands).is_empty());
            EXPECT_TRUE(Internal::allocate_level1_scratch(automatic.recorder(), anchor, 0, 8,
                                                          operands).is_empty());
            EXPECT_TRUE(operands.write_regions().empty());
        }
    } // namespace Test
} // namespace SushiBLAS
