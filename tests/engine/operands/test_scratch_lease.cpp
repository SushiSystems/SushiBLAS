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
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file test_scratch_lease.cpp
 * @brief Tests when a BLAS call leases scratch, how much, and that the lease is declared written.
 * @author Mustafa Garip
 */

#include <complex>
#include <cstddef>
#include <gtest/gtest.h>
#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/engine/execution_mode.hpp>
#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/operands/scratch_lease.hpp>
#include <SushiBLAS/engine_fixture/sushiblas_test.hpp>
#include <SushiBLAS/graph_probe/graph_probe.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Leases scratch through probes over tensors of the fixture's engine. */
        class Medium_ScratchLease : public SushiBLASTest
        {

        };

        TEST(Small_ScratchLease, AnIntegerSolutionNeedsNoWorkspace)
        {
            EXPECT_EQ(Internal::running_solution_bytes(Core::DataType::Int32), std::size_t{0});
        }

        TEST(Small_ScratchLease, AHalfSolutionIsWidenedToFourBytes)
        {
            // HALF accumulates in FLOAT32, so one widened element is four bytes.
            EXPECT_EQ(Internal::running_solution_bytes(Core::DataType::Half), std::size_t{4});
        }

        TEST_F(Medium_ScratchLease, ALeaseReservesTheRequestedBytes)
        {
            GraphProbe probe;
            const Tensor anchor = engine->create_tensor({4});
            OperandSet operands;

            const ScratchBuffer scratch =
                Internal::allocate_blas_scratch(probe.recorder(), anchor, 128, operands);

            EXPECT_EQ(scratch.bytes(), std::size_t{128});
        }

        TEST_F(Medium_ScratchLease, ALeaseIsDeclaredAsOneWrite)
        {
            GraphProbe probe;
            const Tensor anchor = engine->create_tensor({4});
            OperandSet operands;

            (void)Internal::allocate_blas_scratch(probe.recorder(), anchor, 128, operands);

            EXPECT_EQ(operands.write_regions().size(), std::size_t{1});
        }

        TEST_F(Medium_ScratchLease, ForceReferenceLeasesNothing)
        {
            GraphProbe reference(ExecutionMode::ForceReference);
            const Tensor anchor = engine->create_tensor({4});
            OperandSet operands;

            EXPECT_TRUE(Internal::allocate_blas_scratch(reference.recorder(), anchor, 128,
                                                        operands).is_empty());
            EXPECT_TRUE(operands.write_regions().empty());
        }

        TEST_F(Medium_ScratchLease, AZeroByteRequestLeasesNothing)
        {
            GraphProbe automatic;
            const Tensor anchor = engine->create_tensor({4});
            OperandSet operands;

            EXPECT_TRUE(Internal::allocate_blas_scratch(automatic.recorder(), anchor, 0,
                                                        operands).is_empty());
            EXPECT_TRUE(operands.write_regions().empty());
        }

        TEST_F(Medium_ScratchLease, ALevelOneRequestWithNoPartialsLeasesNothing)
        {
            GraphProbe automatic;
            const Tensor anchor = engine->create_tensor({4});
            OperandSet operands;

            EXPECT_TRUE(Internal::allocate_level1_scratch(automatic.recorder(), anchor, 0, 8,
                                                          operands).is_empty());
            EXPECT_TRUE(operands.write_regions().empty());
        }

        TEST_F(Medium_ScratchLease, ALevelOneLeaseIsThePlannedPartialsTimesThePartialBytes)
        {
            GraphProbe probe;
            const Tensor x = engine->create_tensor({1024});
            OperandSet operands;
            constexpr std::size_t K_PARTIAL_BYTES = 8;

            const ScratchBuffer scratch =
                Internal::allocate_level1_scratch(probe.recorder(), x, 1024, K_PARTIAL_BYTES, operands);

            EXPECT_EQ(scratch.bytes(),
                      Reduce::planned_partials(Dispatch::profile_of(x), 1024) * K_PARTIAL_BYTES);
            EXPECT_EQ(operands.write_regions().size(), std::size_t{1});
        }
    } // namespace Test
} // namespace SushiBLAS
