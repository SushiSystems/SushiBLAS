/**************************************************************************/
/* test_primitive_recorder.cpp                                            */
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
 * @file test_primitive_recorder.cpp
 * @brief Tests which types declare themselves an adapter of the PrimitiveRecorder port.
 * @author Mustafa Garip
 * @see tests/foundation/graph_primitives/README.md
 */

#include <gtest/gtest.h>

#include <SushiRuntime/api/graph/recorder.hpp>

#include <SushiBLAS/graph_primitives/host_primitive_recorder.hpp>
#include <SushiBLAS/graph_primitives/primitive_recorder.hpp>
#include <SushiBLAS/graph_primitives/runtime_primitive_recorder.hpp>

namespace SushiBLAS
{
    namespace Test
    {
        /** @brief Tests that the host adapter declares itself a PrimitiveRecorder. */
        TEST(Small_PrimitiveRecorder, TheHostAdapterDeclaresItself)
        {
            EXPECT_TRUE(GraphPrimitives::K_IS_PRIMITIVE_RECORDER<GraphPrimitives::HostPrimitiveRecorder>);
        }

        /** @brief Tests that the runtime adapter declares itself a PrimitiveRecorder. */
        TEST(Small_PrimitiveRecorder, TheRuntimeAdapterDeclaresItself)
        {
            EXPECT_TRUE(
                GraphPrimitives::K_IS_PRIMITIVE_RECORDER<GraphPrimitives::RuntimePrimitiveRecorder>);
        }

        /** @brief Tests that a Recorder with the same members is not taken for an adapter. */
        TEST(Small_PrimitiveRecorder, ARuntimeRecorderIsNotAnAdapter)
        {
            EXPECT_FALSE(GraphPrimitives::K_IS_PRIMITIVE_RECORDER<SushiRuntime::Recorder>);
        }
    } // namespace Test
} // namespace SushiBLAS
